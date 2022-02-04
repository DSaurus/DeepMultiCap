from torch.utils.data import Dataset
import numpy as np
import os
import random
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageDraw
import cv2
import torch
from PIL.ImageFilter import GaussianBlur, MinFilter
import trimesh
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import time
from lib.geometry import *
from lib.sample_util import *
from lib.mesh_util import *
from lib.train_util import find_border


log = logging.getLogger('trimesh')
log.setLevel(40)


class DMCDataset(Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt, cache_data, cache_data_lock, phase='train', yaw_list=range(0, 360, 6), num_views=None):
        self.opt = opt
        self.projection_mode = 'percpective'
        self.phase = phase
        # Path setup
        self.root = self.opt.dataroot
        self.RENDER = os.path.join(self.root, 'img')
        self.PARAM = os.path.join(self.root, 'parameter')
        self.OBJ = os.path.join(self.root, 'obj')
        self.PTS = os.path.join(self.root, 'pts')
        self.SMPL = os.path.join(self.root, 'smplx')
        self.DEPTH = os.path.join(self.root, 'depth')
        self.NORMAL = os.path.join(self.root, 'normal')
        self.SMPL_NORMAL = os.path.join(self.root, 'smpl_pos')
        self.SMPL_POS = os.path.join(self.root, 'smpl_pos_fb')
        self.MASK = os.path.join(self.root, 'mask')
        self.TEX = os.path.join(self.root, 'texture')
        self.COLOR = os.path.join(self.root, 'color')

        if opt.obj_path is not None:
            self.OBJ = opt.obj_path
        if opt.smpl_path is not None:
            self.SMPL = opt.smpl_path
        if opt.tex_path is not None:
            self.TEX = opt.tex_path

        self.smpl_faces = readobj(opt.smpl_faces)['f']

        # self.B_MIN = np.array(self.opt.b_min)
        # self.B_MAX = np.array(self.opt.b_max)

        self.is_train = (phase == 'train')
        self.phase = phase
        self.load_size = self.opt.loadSize

        self.num_views = self.opt.num_views
        if not (num_views is None):
            self.num_views = num_views

        self.num_sample_inout = self.opt.num_sample_inout

        self.yaw_list = yaw_list

        self.select_views_list = {}
        self.subjects = self.get_subjects()
        self.cache_data = cache_data
        self.cache_data_lock = cache_data_lock

        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # augmentation
        self.aug_trans = transforms.Compose([
            transforms.ColorJitter(brightness=opt.aug_bri, contrast=opt.aug_con, saturation=opt.aug_sat,
                                   hue=opt.aug_hue)
        ])

    def clear_cache(self):
        self.cache_data.clear()

    def get_subjects(self):
        all_subjects = os.listdir(self.RENDER)
        
        if os.path.isfile(os.path.join(self.root, 'val.txt')):
            var_subjects = np.loadtxt(os.path.join(self.root, 'val.txt'), dtype=str)
        else:
            var_subjects = []
        if len(var_subjects) == 0:
            return all_subjects
        # return sorted(list(set(all_subjects) - set(var_subjects)))
        if self.is_train:
            return sorted(list(set(all_subjects) - set(var_subjects)))
        else:
            return sorted(list(var_subjects))

    def __len__(self):
        return len(self.subjects) * len(self.yaw_list)

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
        # print(subject)
        view_ids = [self.yaw_list[(yid + len(self.yaw_list) // num_views * offset) % len(self.yaw_list)]
                    for offset in range(num_views)]
        if self.is_train:
            view_ids = np.random.choice(self.yaw_list, num_views, replace=False)
        if self.opt.random_fix_view:
            np.random.seed(int(time.time()))
            if np.random.rand() > 0.5:
                view_ids = [np.random.choice(range(i*360//num_views, (i+1)*360//num_views), 1)[0] for i in range(num_views)]
                # print(view_ids)
        calib_list = []
        render_list = []
        extrinsic_list = []
        depth_list = []
        smpl_norm_list = []
        normal_list = []
        mask_list = []
        ero_mask_list = []

        for vid in view_ids:
            extrinsic_path = os.path.join(self.PARAM, subject, '{}_extrinsic.npy'.format(vid))
            intrinsic_path = os.path.join(self.PARAM, subject, '{}_intrinsic.npy'.format(vid))
            render_path = os.path.join(self.RENDER, subject, '{}.jpg'.format(vid))
            if not os.path.exists(render_path):
                render_path = os.path.join(self.RENDER, subject, '{}.png'.format(vid))
            depth_path = os.path.join(self.DEPTH, subject, '{}.npz'.format(vid))
            normal_path = os.path.join(self.NORMAL, subject, '{}.png'.format(vid))
            if not os.path.exists(normal_path):
                normal_path = os.path.join(self.NORMAL, subject, '{}.jpg'.format(vid))
            smpl_norm_path = os.path.join(self.SMPL_NORMAL, subject, '{}.jpg'.format(vid))
            
            mask_path = os.path.join(self.MASK, subject, '{}.jpg'.format(vid))
            if not os.path.exists(mask_path):
                mask_path = os.path.join(self.MASK, subject, '{}.png'.format(vid))

            # loading calibration data
            extrinsic = np.load(extrinsic_path)
            intrinsic = np.load(intrinsic_path)

            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('RGB')
            else:
                print('warning: no mask')
                mask = render.copy()
            render = Image.open(render_path).convert('RGB')
            if os.path.exists(normal_path):
                normal = Image.open(normal_path)
            else:
                print('warning: no normal')
                normal = render.copy()

            if os.path.exists(depth_path):
                depth = Image.fromarray(np.load(depth_path)['arr_0'])
            else:
                print('warning: no depth')
                depth = render.copy()
            depth = depth.resize((512, 512), Image.BILINEAR)

            if os.path.exists(smpl_norm_path):
                smpl_norm = Image.open(smpl_norm_path)
            else:
                print('warning: no smpl normal')
                smpl_norm = render.copy()

            imgs_list = [render, depth, normal, mask, smpl_norm]
            if self.opt.flip_x:
                for i, img in enumerate(imgs_list):
                    imgs_list[i] = transforms.RandomHorizontalFlip(p=1.0)(img)
                intrinsic[0, :] *= -1.0
                intrinsic[0, 2] += self.load_size

            if self.opt.infer:
                # intrinsic[0, 2] += 10
                if self.opt.infer_reverse:
                    intrinsic[1, :] *= -1.0
                    intrinsic[1, 2] += self.load_size
                if (not self.opt.no_correct):
                    fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
                    x_min, x_max, y_min, y_max = find_border(imgs_list[3])
                    y_min -= 50
                    y_max += 50
                    y_len = y_max - y_min
                    x_min = (x_max + x_min) // 2 - y_len // 2
                    x_max = x_min + y_len
                    scale = 512.0 / y_len

                    fx = fx * scale
                    fy = fy * scale
                    cx = scale * (cx - x_min)
                    cy = scale * (cy - y_min)
                    intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2] = fx, fy, cx, cy
                    depth = transforms.RandomVerticalFlip(p=1.0)(depth)
            else:
                intrinsic[1, :] *= -1.0
                intrinsic[1, 2] += self.load_size

            if self.is_train:
                # Pad images
                pad_size = int(0.1 * self.load_size)
                for i, img in enumerate(imgs_list):
                    imgs_list[i] = ImageOps.expand(img, pad_size, fill=0)

                w, h = imgs_list[0].size
                th, tw = self.load_size, self.load_size

                # random scale
                if self.opt.random_scale:
                    rand_scale = random.uniform(0.9, 1.1)
                    w = int(rand_scale * w)
                    h = int(rand_scale * h)
                    for i, img in enumerate(imgs_list):
                        imgs_list[i] = img.resize((w, h), Image.BILINEAR)
                    intrinsic[0, 0] *= rand_scale
                    intrinsic[1, 1] *= rand_scale

                # random translate in the pixel space
                if self.opt.random_trans:
                    dx = random.randint(-int(round((w - tw) / 10)),
                                        int(round((w - tw) / 10)))
                    dy = random.randint(-int(round((h - th) / 10)),
                                        int(round((h - th) / 10)))
                else:
                    dx = 0
                    dy = 0

                intrinsic[0, 2] += -dx
                intrinsic[1, 2] += -dy

                x1 = int(round((w - tw) / 2.)) + dx
                y1 = int(round((h - th) / 2.)) + dy

                for i, img in enumerate(imgs_list):
                    imgs_list[i] = img.crop((x1, y1, x1 + tw, y1 + th))

                render, depth, normal, mask, smpl_norm = imgs_list
                render = self.aug_trans(render)

                # random blur
                if self.opt.aug_blur > 0.00001:
                    blur = GaussianBlur(np.random.uniform(0, self.opt.aug_blur))
                    render = render.filter(blur)
            else:
                if self.opt.infer and (not self.opt.no_correct):
                    for i, img in enumerate(imgs_list):
                        if i == 4:
                            fill_n = (128, 128, 128)
                        else:
                            fill_n = (0, 0, 0)
                        img = ImageOps.expand(img, 256, fill=fill_n)
                        imgs_list[i] = (img.crop((x_min + 256, y_min + 256, x_max + 256, y_max + 256))).resize(
                            (512, 512), Image.BILINEAR)
                render, depth, normal, mask, smpl_norm = imgs_list

            if self.opt.mask_part:
                mask_draw = ImageDraw.Draw(mask)
                rand_num = np.random.rand()
                if rand_num > 0.75:
                    mask_num = 8
                elif rand_num > 0.25:
                    mask_num = 4
                else:
                    mask_num = 0
                for i in range(mask_num):
                    x, y = np.random.rand() * 512, np.random.rand() * 512
                    w, h = np.random.rand() * 75 + 25, np.random.rand() * 75 + 25
                    mask_draw.rectangle([x, y, x + w, h + y], fill=(0, 0, 0), outline=(0, 0, 0))
                for i in range(mask_num):
                    x, y = np.random.rand() * 512, np.random.rand() * 512
                    w, h = np.random.rand() * 75 + 25, np.random.rand() * 75 + 25
                    mask_draw.ellipse([x, y, x + w, h + y], fill=(0, 0, 0), outline=(0, 0, 0))
                # print(subject)
                # mask.save('metric/attention_mask.png')
                # exit(0)
            ero_mask = mask.filter(MinFilter(3))

            calib = torch.Tensor(np.matmul(intrinsic, extrinsic)).float()
            extrinsic = torch.Tensor(extrinsic[:, :3]).float()

            mask_new = mask.filter(MinFilter(3))
            mask = torch.sum(torch.FloatTensor((np.array(mask).reshape((512, 512, -1)))), dim=2) / 255
            mask[mask > 1] = 1.0
            ero_mask = torch.FloatTensor(np.array(mask).reshape((512, 512, -1)))[:, :, 0] / 255
            render = self.to_tensor(render) * mask.reshape(1, 512, 512)
            normal = self.to_tensor(normal) * mask.reshape(1, 512, 512)
            smpl_norm = self.to_tensor(smpl_norm)

            mask = torch.sum(torch.FloatTensor((np.array(mask_new).reshape((512, 512, -1)))), dim=2) / 255
            mask[mask > 1] = 1.0

            render_list.append(render)
            calib_list.append(calib)
            extrinsic_list.append(extrinsic)
            depth = np.array(depth)
            if len(depth.shape) >= 3:
                depth_list.append(torch.FloatTensor(depth[:, :, 0]) * mask)
            else:
                depth_list.append(torch.FloatTensor(depth) * mask)
            normal_list.append(normal)
            smpl_norm_list.append(smpl_norm)
            mask_list.append(mask.reshape(1, 512, 512))
            ero_mask_list.append(ero_mask.reshape(1, 512, 512))

        return {
            'image': torch.stack(render_list, dim=0),
            'calib': torch.stack(calib_list, dim=0),
            'extrinsic': torch.stack(extrinsic_list, dim=0),
            'depth': torch.stack(depth_list, dim=0),
            'smpl_normal': torch.stack(smpl_norm_list, dim=0),
            'normal': torch.stack(normal_list, dim=0),
            'mask': torch.stack(mask_list, dim=0),
            'ero_mask': torch.stack(ero_mask_list, dim=0)
        }

    
    def visibility_sample(self, data, depth, calib, mask=None):
        surface_points = data['surface_points']
        sample_points = data['sample_points']
        inside = data['inside']
        # plt.subplot(121)
        # plt.imshow(depth[0])
        # plt.subplot(122)
        # plt.imshow(depth[1])
        # plt.savefig('show/depth.jpg')
        depth = depth.clone().unsqueeze(1)
        if self.opt.visibility_sample:
            surface_points = torch.FloatTensor(surface_points.T)
            xyz = []
            for view in range(self.num_views):
                xyz.append(perspective(surface_points.unsqueeze(0), calib[view, :, :].unsqueeze(0)).unsqueeze(1))
            xyz = torch.cat(xyz, dim=1)
            pts_xy = xyz[:, :, :2, :]
            pts_z = xyz[0, :, 2:, :]

            pts_depth = []
            for view in range(self.num_views):
                pts_depth.append(
                    index(depth[view, :, :, :].unsqueeze(0), pts_xy[:, view, :, :], size=self.opt.loadSize))
            pts_depth = torch.cat(pts_depth, dim=0)
            pts_depth[pts_depth < 1e-6] = 1e6
            pts_depth = 1 / pts_depth

            pts_visibility = (pts_depth - pts_z) > -0.005
            pts_visibility = (torch.sum(pts_visibility, dim=0) > 0).squeeze(0)

            inside_points = []
            outside_points = []
            # vin = torch.FloatTensor(surface_points.T)[pts_visibility, :]
            # save_samples_truncted_prob('show/vis.ply', vin, np.ones((vin.shape[0], 1)))

            n = self.opt.num_sample_inout
            vis_pts = torch.FloatTensor(sample_points)[:4 * n][pts_visibility[:4 * n], :]
            vis_inside = torch.BoolTensor(inside)[:4 * n][pts_visibility[:4 * n]]
            vin = vis_pts[vis_inside, :]
            vout = vis_pts[torch.logical_not(vis_inside), :]
            if len(vin.shape) > 1:
                vin = vin[torch.randperm(vin.shape[0]), :]
                inside_points.append(vin[:min(self.num_sample_inout // 2 * self.num_views, vin.shape[0])])
            if len(vout.shape) > 1:
                vout = vout[torch.randperm(vout.shape[0]), :]
                outside_points.append(vout[:min(self.num_sample_inout // 2 * self.num_views, vout.shape[0])])

            # save_samples_truncted_prob('show/vis_in.ply', vin, np.ones((vin.shape[0], 1)))
            # save_samples_truncted_prob('show/vis_out.ply', vout, np.zeros((vout.shape[0], 1)))

            n_vis_pts = torch.FloatTensor(sample_points)[2 * n:6 * n][torch.logical_not(pts_visibility[2 * n:6 * n]), :]
            n_vis_inside = torch.BoolTensor(inside)[2 * n:6 * n][torch.logical_not(pts_visibility[2 * n:6 * n])]
            vin = n_vis_pts[n_vis_inside, :]
            vout = n_vis_pts[torch.logical_not(n_vis_inside), :]
            if len(vin.shape) > 1:
                vin = vin[torch.randperm(vin.shape[0]), :]
                inside_points.append(vin[:min(self.num_sample_inout // 2, vin.shape[0])])
            if len(vout.shape) > 1:
                vout = vout[torch.randperm(vout.shape[0]), :]
                outside_points.append(vout[:min(self.num_sample_inout // 2, vout.shape[0])])

            # save_samples_truncted_prob('show/n_vis_in.ply', vin, np.ones((vin.shape[0], 1)))
            # save_samples_truncted_prob('show/n_vis_out.ply', vout, np.zeros((vout.shape[0], 1)))
            # exit(0)

            ran_pts = torch.FloatTensor(sample_points)[6 * n:]
            ran_inside = torch.BoolTensor(inside)[6 * n:]
            vin = ran_pts[ran_inside, :]
            vout = ran_pts[torch.logical_not(ran_inside), :]
            if len(vin.shape) > 1:
                vin = vin[torch.randperm(vin.shape[0]), :]
                inside_points.append(vin[:min(self.num_sample_inout // 2, vin.shape[0])])
            if len(vout.shape) > 1:
                vout = vout[torch.randperm(vout.shape[0]), :]
                outside_points.append(vout[:min(self.num_sample_inout // 2, vout.shape[0])])

            inside_points = torch.cat(inside_points, dim=0)
            outside_points = torch.cat(outside_points, dim=0)
            # samples = inside_points.transpose(0, 1)
            # labels = torch.ones((1, inside_points.shape[0]))
            samples = torch.cat([inside_points, outside_points], dim=0).transpose(0, 1)
            labels = torch.cat([torch.ones((1, inside_points.shape[0])), torch.zeros((1, outside_points.shape[0]))], 1)
            ran_idx = torch.randperm(samples.shape[1])[:n]
            samples = samples[:, ran_idx]
            labels = labels[:, ran_idx]
            # save_samples_truncted_prob('show/samples.ply', samples.numpy().T, labels.numpy().T)
            # exit(0)
        else:
            inside_points = sample_points[inside]
            np.random.shuffle(inside_points)
            outside_points = sample_points[np.logical_not(inside)]
            np.random.shuffle(outside_points)

            nin = inside_points.shape[0]
            inside_points = inside_points[
                            :self.num_sample_inout // 2] if nin > self.num_sample_inout // 2 else inside_points
            outside_points = outside_points[
                                :self.num_sample_inout // 2] if nin > self.num_sample_inout // 2 else outside_points[
                                                                                                    :(
                                                                                                            self.num_sample_inout - nin)]

            samples = np.concatenate([inside_points, outside_points], 0).T
            labels = np.concatenate([np.ones((1, inside_points.shape[0])), np.zeros((1, outside_points.shape[0]))],
                                    1)

            samples = torch.Tensor(samples).float()
            labels = torch.Tensor(labels).float()
            if self.opt.debug_data:
                save_samples_truncted_prob('show/samples.ply', samples.numpy().T, labels.numpy().T)
        return {
            'samples': samples,
            'labels': labels,
            'feat_points': data['feat_points']
        }

    def select_sampling_method(self, subject, b_min, b_max):
        if self.cache_data.__contains__(subject):
            return self.cache_data[subject]
        print(subject, self.cache_data.__len__())
        root_dir = self.OBJ
        sub_name = subject
        if self.phase != 'inference':
            mesh = trimesh.load(os.path.join(root_dir, sub_name, '%s.obj' % sub_name))
            if self.opt.coarse_part:
                radius_list = [self.opt.sigma, self.opt.sigma * 2, self.opt.sigma * 4]
            else:
                radius_list = [self.opt.sigma / 3, self.opt.sigma, self.opt.sigma * 2]
            surface_points = np.zeros((6 * self.num_sample_inout, 3))
            sample_points = np.zeros((6 * self.num_sample_inout, 3))
            for i in range(3):
                d = 2 * self.num_sample_inout
                surface_points[i * d:(i + 1) * d, :], _ = trimesh.sample.sample_surface(mesh,
                                                                                        2 * self.num_sample_inout)
                sample_points[i * d:(i + 1) * d, :] = surface_points[i * d:(i + 1) * d, :] + np.random.normal(
                    scale=radius_list[i], size=(2 * self.num_sample_inout, 3))

            # add random points within image space
            length = b_max - b_min
            random_points = np.random.rand(self.num_sample_inout, 3) * length + b_min
            sample_points = np.concatenate([sample_points, random_points], 0)
            inside = mesh.contains(sample_points)

            del mesh
        else:
            sample_points = torch.zeros(1)
            surface_points = torch.zeros(1)
            inside = torch.zeros(1)

        feat_points = torch.zeros(1)

        self.cache_data_lock.acquire()
        self.cache_data[subject] = {
            'sample_points': sample_points,
            'surface_points': surface_points,
            'inside': inside,
            'feat_points': feat_points
        }
        self.cache_data_lock.release()

        return self.cache_data[subject]

    def get_norm(self, subject):
        b_min = torch.zeros(3).float()
        b_max = torch.zeros(3).float()
        scale = torch.zeros(1).float()
        center = torch.zeros(3).float()

        t3_mesh = readobj(os.path.join(self.SMPL, subject, 'smplx.obj'))['vi'][:, :3]
        b0 = np.min(t3_mesh, axis=0)
        b1 = np.max(t3_mesh, axis=0)
        center = torch.FloatTensor((b0 + b1) / 2)
        scale = torch.FloatTensor([np.min(1.0 / (b1 - b0)) * 0.9])
        b_min = center - 0.5 / scale
        b_max = center + 0.5 / scale

        normal = np.zeros((3))
        for f in self.smpl_faces:
            a, b, c = t3_mesh[f[0]][0], t3_mesh[f[1]][0], t3_mesh[f[2]][0]
            normal += cross_3d(c - a, b - a)
        del t3_mesh
        if self.opt.flip_normal:
            normal = -normal

        return {
            'b_min': b_min,
            'b_max': b_max,
            'scale': scale,
            'center': center,
            'direction': normal
        }

    def get_item(self, index):
        
        sid = index % len(self.subjects)
        tmp = index // len(self.subjects)
        yid = tmp % len(self.yaw_list)


        subject = self.subjects[sid]
        res = {
            'name': subject,
            'mesh_path': os.path.join(self.OBJ, subject, subject + '.obj'),
            'sid': sid,
            'yid': yid,
        }
        render_data = self.get_render(subject, num_views=self.num_views, yid=yid,
                                      random_sample=self.opt.random_multiview)
        res.update(render_data)

        norm_parameter = self.get_norm(subject)
        res.update(norm_parameter)

        sample_data = self.select_sampling_method(subject, res['b_min'].numpy(), res['b_max'].numpy())
        if self.phase != 'inference':
            sample_data = self.visibility_sample(sample_data, res['depth'], res['calib'], res['mask'])
        res.update(sample_data)

        mesh = trimesh.load(os.path.join(self.SMPL, subject, 'smplx.obj'))
        res['extrinsic'][0, :, :] = 0
        for i in range(3):
            res['extrinsic'][0, i, i] = 1

        translation = np.zeros((4, 4))
        translation[:3, 3] = -np.array(res['center']) * res['scale'].numpy()
        translation[1, 3] += 0.5
        for i in range(3):
            translation[i, i] = res['scale'].numpy()
        translation[3, 3] = 1
        mesh.apply_transform(translation)
        if self.opt.infer:
            print('after: ', mesh.bounds)

        # center
        transform = np.zeros((4, 4))
        for i in range(4):
            transform[i, i] = 1
        transform[1, 3] = -0.5
        mesh.apply_transform(transform)

        # flip
        # if self.opt.flip_smpl:
        #     res['direction'] = -res['direction']

        # rotation
        direction = res['direction']
        x, z = direction[0], direction[2]
        theta = math.acos(z / math.sqrt(z * z + x * x))
        if x < 0:
            theta = 2 * math.acos(-1) - theta
        res['extrinsic'][0] = torch.FloatTensor(rotationY(-theta))
        if self.opt.flip_smpl:
            res['extrinsic'][0] = res['extrinsic'][0] @ torch.FloatTensor(rotationX(math.acos(-1)))

        if self.opt.random_rotation:
            pi = math.acos(-1)
            beta = 40 * pi / 180
            rand_rot = np.array(rotationX((np.random.rand() - 0.5) * beta)) @ np.array(
                rotationY((np.random.rand() - 0.5) * beta)) @ np.array(rotationZ((np.random.rand() - 0.5) * beta))
            res['extrinsic'][0] = torch.FloatTensor(rand_rot) @ res['extrinsic'][0]

        rotation = np.zeros((4, 4))
        rotation[3, 3] = 1
        rotation[:3, :3] = res['extrinsic'][0]
        mesh.apply_transform(rotation)

        transform[1, 3] = 0.5
        mesh.apply_transform(transform)

        vox = mesh.voxelized(pitch=1.0 / 128, method='binvox', bounds=np.array([[-0.5, 0, -0.5], [0.5, 1, 0.5]]),
                                exact=True)
        
        vox.fill()
        res['vox'] = torch.FloatTensor(vox.matrix).unsqueeze(0)

        if self.opt.debug_data:
            img = np.uint8(
                (np.transpose(render_data['image'][0][0:3, :, :].numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0)
            img = np.array(img, dtype=np.uint8).copy()
            calib = render_data['calib'][0]
            # pts = torch.FloatTensor(res['samples'][:, res['labels'][0] > 0.5]) # [3, N]
            # pts = res['samples']
            pts = torch.FloatTensor(res['feat_points'])
            print(pts)
            pts = perspective(pts.unsqueeze(0), calib.unsqueeze(0)).squeeze(0).transpose(0, 1)
            for p in pts:
                img = cv2.circle(img, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1)
            cv2.imwrite('show/test_test.jpg', img)
            exit(0)

        return res

    def __getitem__(self, index):
        return self.get_item(index)
