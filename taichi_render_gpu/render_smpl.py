import taichi as ti
import taichi_three as t3
import numpy as np
from taichi_three.transform import *
from tqdm import tqdm
import os
import sys
import cv2
import trimesh
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from lib.smpl_util import *
from lib.mesh_util import *

def read_norm_smpl(path, smpl_faces, flip_normal=False, init_rot=None):
    obj = t3.readobj(path, scale=1)
    faces = t3.readobj(smpl_faces)['f']
    o_vi = obj['vi'].copy()
    norm_vi = smpl_normalize(obj, faces, flip_normal, init_rot)['smpl']
    obj['vi'] = norm_vi
    vn = calc_smpl_normal(obj)
    obj['vi'] = o_vi
    obj['vp'] = norm_vi
    obj['vn'] = vn
    return obj
            

def render_smpl_global_normal(dataroot, obj_path, faces_path, res=(1024, 1024), angles=range(360), flip_y=False, flip_normal=False, init_rot=None):
    ti.init(ti.cpu)
    pos_save_root = os.path.join(dataroot, 'smpl_pos')
    img_save_root = os.path.join(dataroot, 'smpl_render')
    os.makedirs(pos_save_root, exist_ok=True)
    os.makedirs(img_save_root, exist_ok=True)
    img_path = os.path.join(dataroot, 'img')
    parameter_path = os.path.join(dataroot, 'parameter')
    obj_list = os.listdir(obj_path)
    obj = read_norm_smpl(os.path.join(obj_path, obj_list[0], 'smplx.obj'), faces_path, flip_normal, init_rot)
    model = t3.Model(obj=obj, col_n=obj['vi'].shape[0])
    
    scene = t3.Scene()
    scene.add_model(model)
    light_dir = np.array([0, 0, 1])
    for l in range(4):
        rotate = np.matmul(rotationX(math.radians(np.random.uniform(-30, 30))),
                           rotationY(math.radians(360 // 4 * l)))
        dir = [*np.matmul(rotate, light_dir)]
        light = t3.Light(dir, color=[1.0, 1.0, 1.0])
        scene.add_light(light)

    camera = t3.Camera(res=res)
    scene.add_camera(camera)
    scene.init()
    for obj_name in tqdm(os.listdir(parameter_path)):
        pos_save_path = os.path.join(pos_save_root, obj_name)
        img_save_path = os.path.join(img_save_root, obj_name)
        os.makedirs(img_save_path, exist_ok=True)
        if os.path.exists(pos_save_path) and len(os.listdir(os.path.join(pos_save_path))) == len(angles):
            continue
        if not os.path.exists(os.path.join(obj_path, obj_name, 'smplx.obj')):
            continue
        obj = read_norm_smpl(os.path.join(obj_path, obj_name, 'smplx.obj'), faces_path, flip_normal, init_rot)
        os.makedirs(pos_save_path, exist_ok=True)
        for angle in angles:
            intrinsic = np.load(os.path.join(parameter_path, obj_name, '{}_intrinsic.npy'.format(angle)))
            extrinsic = np.load(os.path.join(parameter_path, obj_name, '{}_extrinsic.npy'.format(angle)))
            
            if flip_y:
                camera.set_intrinsic(fx=intrinsic[0, 0], fy=-intrinsic[1, 1], cx=intrinsic[0, 2], cy=res[0]-intrinsic[1, 2])
            else:    
                camera.set_intrinsic(fx=intrinsic[0, 0], fy=intrinsic[1, 1], cx=intrinsic[0, 2], cy=intrinsic[1, 2])

            trans = extrinsic[:, :3]
            T = extrinsic[:, 3]
            p = -trans.T @ T
            camera.set_extrinsic(trans.T, p)
            color = obj['vn']
            model.from_obj(obj)
            model.vc.from_numpy(color)
            model.type[None] = 1
            camera._init()
            scene.render()
            
            ti.imwrite( (camera.img.to_numpy() + 1)/2, os.path.join(pos_save_path, '{}.jpg'.format(angle)))


if __name__ == '__main__':
    res = (512, 512)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str)
    parser.add_argument("--obj_path", type=str)
    parser.add_argument("--faces_path", type=str)
    parser.add_argument("--yaw_list", type=int, nargs='+', default=[i for i in range(360)])
    parser.add_argument("--flip_y", action="store_true")
    parser.add_argument("--flip_normal", action="store_true")
    args = parser.parse_args()

    render_smpl_global_normal(args.dataroot, args.obj_path, args.faces_path, res, args.yaw_list, args.flip_y, args.flip_normal)
    
