from torch.utils.data import Dataset
from multiprocessing import Process, Manager, Lock
import numpy as np
import os
import random
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import cv2
import torch
from PIL.ImageFilter import GaussianBlur
import trimesh
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

class NormalDataset(Dataset):
    def __init__(self, opt, phase = 'train'):
        self.opt = opt

        self.root = opt.dataroot
        self.NORMAL = os.path.join(self.root, 'normal')
        self.RENDER = os.path.join(self.root, 'img')
        self.MASK = os.path.join(self.root, 'mask')
        self.phase = phase
        self.is_train = self.phase == 'train'
        self.load_size = 512

        self.subjects = self.get_subjects()
        self.yaw_list = range(0, 360, 6)

        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.Resize(512),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # augmentation
        self.aug_trans = transforms.Compose([
            transforms.ColorJitter(brightness=opt.aug_bri, contrast=opt.aug_con, saturation=opt.aug_sat,
                                   hue=opt.aug_hue)
        ])
    
    
    def __len__(self):
        return len(self.subjects) * len(self.yaw_list)

    def get_subjects(self):
        all_subjects = os.listdir(self.RENDER)
        if os.path.isfile(os.path.join(self.root, 'val.txt')):
            var_subjects = np.loadtxt(os.path.join(self.root, 'val.txt'), dtype=str)
        else:
            var_subjects = []
        if len(var_subjects) == 0:
            return all_subjects

        if self.is_train:
            return sorted(list(set(all_subjects) - set(var_subjects)))
        else:
            return sorted(list(var_subjects)) 
    
    def get_render(self, subject, num_views, yid=0, random_sample=False):
        '''
        Return the render data
        :param subject: subject name
        :param num_views: how many views to return
        :param view_id: the first view_id. If None, select a random one.
        :return:
            'img': [num_views, C, W, H] images
            'calib': [num_views, 4, 4] calibration matrix
            'extrinsic': [num_views, 4, 4] extrinsic matrix
            'mask': [num_views, 1, W, H] masks
        '''

        # The ids are an even distribution of num_views around view_id
        view_ids = [self.yaw_list[(yid + len(self.yaw_list) // num_views * offset) % len(self.yaw_list)]
                    for offset in range(1)]
        if random_sample and self.is_train:
            view_ids = np.random.choice(self.yaw_list, num_views, replace=False)

        render_list = []
        normal_list = []
        mask_list = []

        for vid in view_ids:
            render_path = os.path.join(self.RENDER, subject, '{}.jpg'.format(vid))
            if not os.path.exists(render_path):
                render_path = os.path.join(self.RENDER, subject, '{}.png'.format(vid))
            normal_path = os.path.join(self.NORMAL, subject, '{}.png'.format(vid))
            if not os.path.exists(normal_path):
                normal_path = os.path.join(self.NORMAL, subject, '{}.jpg'.format(vid))
            mask_path = os.path.join(self.MASK, subject, '{}.jpg'.format(vid))
            if not os.path.exists(mask_path):
                mask_path = os.path.join(self.MASK, subject, '{}.png'.format(vid))
                
            mask = Image.open(mask_path)
            render = Image.open(render_path).convert('RGB')
            normal = Image.open(normal_path).convert('RGB')
            
            if self.is_train:
                # Pad images
                pad_size = int(0.1 * self.load_size)
                render = ImageOps.expand(render, pad_size, fill=0)
                normal = ImageOps.expand(normal, pad_size, fill=0)
                mask = ImageOps.expand(mask, pad_size, fill=0)

                w, h = render.size
                th, tw = self.load_size, self.load_size

                # random scale
                if self.opt.random_scale:
                    rand_scale = random.uniform(0.9, 1.1)
                    w = int(rand_scale * w)
                    h = int(rand_scale * h)
                    render = render.resize((w, h), Image.BILINEAR)                    
                    normal = normal.resize((w, h), Image.BILINEAR)
                    mask = mask.resize((w, h), Image.BILINEAR)

                # random translate in the pixel space
                if self.opt.random_trans:
                    dx = random.randint(-int(round((w - tw) / 10)),
                                        int(round((w - tw) / 10)))
                    dy = random.randint(-int(round((h - th) / 10)),
                                        int(round((h - th) / 10)))
                else:
                    dx = 0
                    dy = 0

                x1 = int(round((w - tw) / 2.)) + dx
                y1 = int(round((h - th) / 2.)) + dy

                render = render.crop((x1, y1, x1 + tw, y1 + th))
                normal = normal.crop((x1, y1, x1 + tw, y1 + th))
                mask = mask.crop((x1, y1, x1 + tw, y1 + th))

                render = self.aug_trans(render)

                # random blur
                if self.opt.aug_blur > 0.00001:
                    blur = GaussianBlur(np.random.uniform(0, self.opt.aug_blur))
                    render = render.filter(blur)
            
            mask = torch.FloatTensor(np.array(mask))[:, :, 0] / 255
            render = self.to_tensor(render) * mask.reshape(1, 512, 512)
            normal = self.to_tensor(normal) * mask.reshape(1, 512, 512)

            render_list.append(render)
            normal_list.append(normal)
            mask_list.append(mask.reshape(1, 512, 512))

        return {
            'img': torch.stack(render_list, dim=0),
            'normal': torch.stack(normal_list, dim=0),
            'mask': torch.stack(mask_list, dim=0)
        }
    
    def get_item(self, index):
        sid = index % len(self.subjects)
        tmp = index // len(self.subjects)
        yid = tmp % len(self.yaw_list)

        subject = self.subjects[sid]
        res = {
            'name': subject,
            'sid': sid,
            'yid': yid,
        }
        render_data = self.get_render(subject, num_views=1, yid=yid,
                                      random_sample=self.opt.random_multiview)
        res.update(render_data)
        return res

    def __getitem__(self, index):
        return self.get_item(index)
    