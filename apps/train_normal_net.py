import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import threading
import time
import json
import numpy as np
import cv2
import random
import torch
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import matplotlib.pyplot as plt

from lib.mesh_util import *
from lib.sample_util import *
from lib.train_util import *
from lib.data import *
from lib.model import *
from lib.geometry import index
from lib.loss_util import VGGPerceptualLoss
from lib.options import parse_config

# get options
opt = parse_config()
log = SummaryWriter(opt.log_path)

def train(opt):
    
    gpu_ids = [int(i) for i in opt.gpu_ids.split(',')]
    cuda = torch.device("cuda:%d" % gpu_ids[0])
    netN = NormalNet()
    dataset = NormalDataset(opt)
    train_data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=not opt.serial_batches,
                                   num_workers=opt.num_threads, pin_memory=opt.pin_memory)

    netN.to(cuda)
    netN = DataParallel(netN, device_ids=gpu_ids)
    
    if opt.load_netN_checkpoint_path is not None:
        netN.load_state_dict(torch.load(opt.load_netN_checkpoint_path), strict=False)

    lr = opt.learning_rate
    optimizerN = torch.optim.Adam(netN.parameters(), lr=lr)

    os.makedirs(opt.checkpoints_path, exist_ok=True)
    os.makedirs(opt.results_path, exist_ok=True)
    os.makedirs('%s/%s' % (opt.checkpoints_path, opt.name), exist_ok=True)
    os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)

    EPOCH = 100
    total_iteration = 0
    
    perceptual_loss = VGGPerceptualLoss().to(cuda)
    for epoch in range(EPOCH):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        train_bar = tqdm(enumerate(train_data_loader))
        for train_idx, train_data in train_bar:
            
            total_iteration += 1
            iter_start_time = time.time()

            # retrieve the data
            image_tensor = train_data['img'].to(device=cuda)
            normal_tensor = train_data['normal'].to(device=cuda)
            mask_tensor = train_data['mask'].to(device=cuda)

            res = netN.forward(image_tensor)
            res = res * mask_tensor
            error = F.l1_loss(normal_tensor, res)
            perceptual_error = perceptual_loss(normal_tensor.squeeze(1), res.squeeze(1))

            error = 5*error + perceptual_error
            
            optimizerN.zero_grad()
            error.backward()
            optimizerN.step()

            iter_net_time = time.time()
            eta = ((iter_net_time - epoch_start_time) / (train_idx + 1)) * len(train_data_loader) - (
                    iter_net_time - epoch_start_time)

            log.add_scalar('loss', error.item(), total_iteration)
            if train_idx % opt.freq_plot == 0:
                descrip = 'Name: {0} | Epoch: {1} | {2}/{3} | Err: {4:.06f} | LR: {5:.06f} | Sigma: {6:.02f} | dataT: {7:.05f} | netT: {8:.05f} | ETA: {9:02d}:{10:02d}'.format(
                    opt.name, epoch, train_idx, len(train_data_loader), error.item(), lr, opt.sigma,
                    iter_start_time - iter_data_time,
                    iter_net_time - iter_start_time, int(eta // 60),
                    int(eta - 60 * (eta // 60)))
                train_bar.set_description(descrip)

            if train_idx % opt.freq_save == 0 and train_idx != 0:
                torch.save(netN.state_dict(), '%s/%s/netN_latest' % (opt.checkpoints_path, opt.name))
                torch.save(netN.state_dict(), '%s/%s/netN_epoch_%d' % (opt.checkpoints_path, opt.name, epoch))
            
            if train_idx % opt.freq_normal_show == 0:
                show_img = (image_tensor[0][0].cpu().detach().permute(1, 2, 0) + 0.5).numpy()
                net_normal = (res[0][0].cpu().detach().permute(1, 2, 0) + 0.5).numpy()
                gt_normal = (normal_tensor[0][0].cpu().detach().permute(1, 2, 0) + 0.5).numpy()
                plt.subplot(131)
                plt.imshow(show_img)
                plt.subplot(132)
                plt.imshow(net_normal)
                plt.subplot(133)
                plt.imshow(gt_normal)
                plt.savefig('%s/%s/epoch%03d_%05d.jpg' % (opt.results_path, opt.name, epoch, train_idx))
                plt.close('all')

            iter_data_time = time.time()

if __name__ == "__main__":
    train(opt)