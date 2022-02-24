import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import time
import json
import numpy as np
import cv2
import random
import torch
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from multiprocessing import Process, Manager, Lock

from lib.options import parse_config
from lib.mesh_util import *
from lib.sample_util import *
from lib.train_util import *
from lib.data import *
from lib.model import *
from lib.geometry import index

import matplotlib
matplotlib.use('AGG')

# get options
opt = parse_config()

def inference_nm(opt):
    # set cuda
    cuda = torch.device('cuda:%s' % opt.gpu_ids[0])
    netG = DMCNet(opt, projection_mode='perspective').to(device=cuda)
    netN = NormalNet().to(cuda)
    print('Using Network: ', netG.name)
    gpu_ids = [int(i) for i in opt.gpu_ids.split(',')]
    netG = DataParallel(netG, device_ids=gpu_ids)
    netN = DataParallel(netN, device_ids=gpu_ids)
    # load checkpoints
    if opt.load_netG_checkpoint_path is not None:
        print('loading for net G ...', opt.load_netG_checkpoint_path)
        netG.load_state_dict(torch.load(opt.load_netG_checkpoint_path, map_location=cuda), strict=False)
    
    if opt.load_netN_checkpoint_path is not None:
        print('loading for net N ...', opt.load_netN_checkpoint_path)
        netN.load_state_dict(torch.load(opt.load_netN_checkpoint_path, map_location=cuda), strict=False)
    
    print("loaded finished!")

    test_netG = netG.module
    netN = netN.module
    dataset = DMCDataset(opt, phase='inference', yaw_list=opt.yaw_list, cache_data = Manager().dict(), cache_data_lock=Lock())
    print(dataset.__len__())
    
    os.makedirs(opt.results_path, exist_ok=True)
    os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)

    opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
    with open(opt_log, 'w') as outfile:
        outfile.write(json.dumps(vars(opt), indent=2))

    with torch.no_grad():
        test_netG.eval()
        dic = {}
        print('generate mesh (inference) ...')
        for i in tqdm(range(0, len(dataset) // opt.num_views)):
            test_data = dataset[i]
            save_path = '%s/%s/inference_eval_%s_%d.obj' % (
                opt.results_path, opt.name, test_data['name'], test_data['yid'])
            if dic.__contains__(test_data['name']):
                break
            dic[test_data['name']] = True
            image_tensor = test_data['image'].to(device=cuda).unsqueeze(0)
            mask_tensor = test_data['mask'].to(device=cuda).unsqueeze(0)
            res = netN.forward(image_tensor)
            res = res * mask_tensor
            # test_data['normal'] = torch.nn.Upsample(size=[1024, 1024], mode='bilinear')(res[0])
            test_data['normal'] = res[0]
            print('saving to ' + save_path)
            gen_mesh_dmc(opt, test_netG, cuda, test_data, save_path, threshold=opt.mc_threshold, use_octree=True)


if __name__ == '__main__':
    np.random.seed(int(time.time()))
    random.seed(int(time.time()))
    torch.manual_seed(int(time.time()))

    inference_nm(opt)
   