import torch
import numpy as np
from .mesh_util import *
from .sample_util import *
from .geometry import *
import cv2
from PIL import Image
from tqdm import tqdm
from PIL.ImageFilter import MinFilter

def find_border(img):
    img = img.filter(MinFilter(11))
    img = np.array(img)
    img_1 = np.sum(img, axis=2)
    img_x = np.sum(img_1, axis=0)
    img_y = np.sum(img_1, axis=1)
    x_min = img_x.shape[0]
    x_max = 0
    y_min = img_y.shape[0]
    y_max = 0
    for x in range(img_x.shape[0]):
        if img_x[x] > 0:
            x_min = x
            break
    for x in range(img_x.shape[0] - 1, 0, -1):
        if img_x[x] > 0:
            x_max = x
            break
    for y in range(img_y.shape[0]):
        if img_y[y] > 0:
            y_min = y
            break
    for y in range(img_y.shape[0] - 1, 0, -1):
        if img_y[y] > 0:
            y_max = y
            break
    return x_min, x_max, y_min, y_max

def reshape_multiview_tensors(image_tensor, calib_tensor):
    # Careful here! Because we put single view and multiview together,
    # the returned tensor.shape is 5-dim: [B, num_views, C, W, H]
    # So we need to convert it back to 4-dim [B*num_views, C, W, H]
    # Don't worry classifier will handle multi-view cases
    image_tensor = image_tensor.view(
        image_tensor.shape[0] * image_tensor.shape[1],
        image_tensor.shape[2],
        image_tensor.shape[3],
        image_tensor.shape[4]
    )
    calib_tensor = calib_tensor.view(
        calib_tensor.shape[0] * calib_tensor.shape[1],
        calib_tensor.shape[2],
        calib_tensor.shape[3]
    )

    return image_tensor, calib_tensor

def reshape_multiview_tensors_3d(image_tensor, calib_tensor, extrinsic):
    # Careful here! Because we put single view and multiview together,
    # the returned tensor.shape is 5-dim: [B, num_views, C, W, H]
    # So we need to convert it back to 4-dim [B*num_views, C, W, H]
    # Don't worry classifier will handle multi-view cases
    image_tensor = image_tensor.view(
        image_tensor.shape[0] * image_tensor.shape[1],
        image_tensor.shape[2],
        image_tensor.shape[3],
        image_tensor.shape[4]
    )
    calib_tensor = calib_tensor.view(
        calib_tensor.shape[0] * calib_tensor.shape[1],
        calib_tensor.shape[2],
        calib_tensor.shape[3]
    )
    extrinsic = extrinsic.view(
        extrinsic.shape[0] * extrinsic.shape[1],
        extrinsic.shape[2],
        extrinsic.shape[3]
    )

    return image_tensor, calib_tensor, extrinsic


def reshape_sample_tensor(sample_tensor, num_views):
    if num_views == 1:
        return sample_tensor
    # Need to repeat sample_tensor along the batch dim num_views times
    sample_tensor = sample_tensor.unsqueeze(dim=1)
    sample_tensor = sample_tensor.repeat(1, num_views, 1, 1)
    sample_tensor = sample_tensor.view(
        sample_tensor.shape[0] * sample_tensor.shape[1],
        sample_tensor.shape[2],
        sample_tensor.shape[3]
    )
    return sample_tensor


def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr