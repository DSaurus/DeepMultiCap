import taichi as ti
import taichi_three as t3
import numpy as np
from taichi_three.transform import *
from tqdm import tqdm
import os
import time
import cv2
import json
os.environ["KMP_DUPLICATE_LIB_OK"]="True"

def find_border(img):
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
    for x in range(img_x.shape[0]-1, 0, -1):
        if img_x[x] > 0:
            x_max = x
            break
    for y in range(img_y.shape[0]):
        if img_y[y] > 0:
            y_min = y
            break
    for y in range(img_y.shape[0]-1, 0, -1):
        if img_y[y] > 0:
            y_max = y
            break
    return x_min, x_max, y_min, y_max

class StaticRenderer:
    def __init__(self):
        ti.init(ti.cpu)
        self.scene = t3.Scene()
        self.N = 10
    
    def change_all(self):
        save_obj = []
        save_tex = []
        for model in self.scene.models:
            save_obj.append(model.init_obj)
            save_tex.append(model.init_tex)
        ti.init(ti.cpu)
        print('init')
        self.scene = t3.Scene()
        for i in range(len(save_obj)):
            model = t3.StaticModel(self.N, obj=save_obj[i], tex=save_tex[i])
            self.scene.add_model(model)

    def check_update(self, obj):
        temp_n = self.N
        self.N = max(obj['vi'].shape[0], self.N)
        self.N = max(obj['f'].shape[0], self.N)
        if not (obj['vt'] is None):
            self.N = max(obj['vt'].shape[0], self.N)

        if self.N > temp_n:
            self.N *= 2
            self.change_all()
            self.camera_light()
    
    def add_model(self, obj, tex=None):
        self.check_update(obj)
        model = t3.StaticModel(self.N, obj=obj, tex=tex)
        self.scene.add_model(model)
    
    def modify_model(self, index, obj, tex=None):
        self.check_update(obj)
        self.scene.models[index].init_obj = obj
        self.scene.models[index].init_tex = tex
        self.scene.models[index]._init()
    
    def camera_light(self):
        camera = t3.Camera(res=res)
        camera1 = t3.Camera(res=(512, 512))
        self.scene.add_camera(camera)
        self.scene.add_camera(camera1)
        light_dir = np.array([0, 0, 1])
        for l in range(6):
            rotate = np.matmul(rotationX(math.radians(np.random.uniform(-30, 30))),
                            rotationY(math.radians(360 // 6 * l)))
            dir = [*np.matmul(rotate, light_dir)]
            light = t3.Light(dir, color=[1.0, 1.0, 1.0])
            self.scene.add_light(light)

def render_mv_random_mask(renderer, data_path, texture_path, data_id, save_path, res=(1024, 1024), enable_gpu=False, dis_scale=1, 
    ran_mask_num = 0):
    img_path = os.path.join(texture_path, data_id + '.jpg')
    obj_path = os.path.join(data_path, data_id, data_id + '.obj')
    obj_names = os.listdir(data_path)
    
    img_save_path = os.path.join(save_path, 'img', data_id)
    depth_save_path = os.path.join(save_path, 'depth', data_id)
    normal_save_path = os.path.join(save_path, 'normal', data_id)
    mask_save_path = os.path.join(save_path, 'mask', data_id)
    parameter_save_path = os.path.join(save_path, 'parameter', data_id)
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)
    if not os.path.exists(parameter_save_path):
        os.makedirs(parameter_save_path)
    if not os.path.exists(mask_save_path):
        os.makedirs(mask_save_path)
    if not os.path.exists(normal_save_path):
        os.makedirs(normal_save_path)
    if not os.path.exists(depth_save_path):
        os.makedirs(depth_save_path)

    texture = ti.imread(img_path)
    obj = t3.readobj(obj_path, scale=1)
    
    if len(renderer.scene.models) >= 1:
        renderer.modify_model(0, obj, texture)
    else:
        renderer.add_model(obj, texture)
    
    mask_shapes = []
    mask_objs = []
    for i in range(ran_mask_num):
        mask_name = np.random.choice(obj_names, 1)[0]
        mask_obj_path = os.path.join(data_path, mask_name, mask_name + '.obj')
        mask_tex_path = os.path.join(texture_path, mask_name + '.jpg')

        msk_tex = ti.imread(mask_tex_path)
        msk_obj = t3.readobj(mask_obj_path, scale=1)
        rotation = np.array(rotationY(math.radians(np.random.uniform(0, 360))))
        msk_obj['vi'][:, :3] = (rotation @ msk_obj['vi'][:, :3].T).T
        obj_bbx = np.max(obj['vi'], axis=0) - np.min(obj['vi'], axis=0)
        mask_obj_bbx = np.max(msk_obj['vi'], axis=0) - np.min(msk_obj['vi'], axis=0)
        r = np.sqrt( (obj_bbx[0] + mask_obj_bbx[0])**2 + (obj_bbx[2] + mask_obj_bbx[2])**2) / 2 
        ran = np.random.rand() * math.acos(-1) * 2
        shift = np.array([r*math.sin(ran), 0, r*math.cos(ran)])
        msk_obj['vi'][:, :3] += shift.reshape((1, 3))
        if len(renderer.scene.models) >= 2+i:
            renderer.modify_model(i+1, msk_obj, msk_tex)
        else:
            renderer.add_model(msk_obj, msk_tex)
        mask_shapes.append(msk_obj['vi'].shape[0])
        mask_objs.append(msk_obj)

    vi = obj['vi']
    median = np.median(vi, axis=0)  # + (np.random.randn(3) - 0.5) * 0.2
    vmin = vi.min(0)
    vmax = vi.max(0)
    median[1] = (vmax[1] * 4 + vmin[1] * 3) / 7

    r_color = np.zeros((obj['vi'].shape[0], 3))
    r_color[:, 0] = 1
    renderer.scene.models[0].modify_color(r_color)
    for i in range(1, ran_mask_num+1):
        b_color = np.zeros((mask_shapes[i-1], 3))
        b_color[:, 2] = 1
        renderer.scene.models[i].modify_color(b_color)

    angle_mul = 1
    for angle in tqdm(range(360), desc='angle'):
        for i in range(ran_mask_num + 1):
            renderer.scene.models[i].type[None] = 0
        # if (os.path.exists(os.path.join(img_save_path, '{}.jpg'.format(angle)))):
        #     continue
        dis = vmax[1] - vmin[1]
        dis *= dis_scale
        ori_vec = np.array([0, 0, dis])
        p = np.random.uniform(-30, 10)
        rotate = np.matmul(rotationY(math.radians(angle*angle_mul)), rotationX(math.radians(p)))
        fwd = np.matmul(rotate, ori_vec)
        fx = res[0] * 0.5
        fy = res[1] * 0.5
        cx = fx
        cy = fy
        target = median
        pos = target + fwd
        renderer.scene.cameras[0].set(pos=pos, target=target)
        renderer.scene.cameras[0].set_intrinsic(fx, fy, cx, cy)
        renderer.scene.cameras[0]._init()
        renderer.scene.single_render(0)

        img = renderer.scene.cameras[0].img.to_numpy()

        x_min, x_max, y_min, y_max = find_border(img)

        x_min -= 20
        x_max += 20
        x_len = x_max - x_min
        y_min = (y_max + y_min - x_len) // 2
        scale = 512.0 / x_len
        fx = 512 * scale
        fy = 512 * scale
        cx = scale * (cx - y_min)
        cy = scale * (cy - x_min)
        renderer.scene.cameras[1].set_intrinsic(fx, fy, cx, cy)
        renderer.scene.cameras[1].set(pos=pos,target=target)
        renderer.scene.cameras[1]._init()
        renderer.scene.single_render(0)
        camera1 = renderer.scene.cameras[1]
        depth_map = camera1.zbuf.to_numpy().swapaxes(0, 1)[::-1, :]
        np.savez(os.path.join(depth_save_path, '{}.npz'.format(angle)), depth_map)

        renderer.scene.render()
        camera1 = renderer.scene.cameras[1]
        np.save(os.path.join(parameter_save_path, '{}_extrinsic.npy'.format(angle)),
                camera1.export_extrinsic())
        np.save(os.path.join(parameter_save_path, '{}_intrinsic.npy'.format(angle)),
                camera1.export_intrinsic())
        ti.imwrite(camera1.img, os.path.join(img_save_path, '{}.png'.format(angle)))
        ti.imwrite(camera1.normal_map, os.path.join(normal_save_path, '{}.png'.format(angle)))
        
        for i in range(ran_mask_num + 1):
            renderer.scene.models[i].type[None] = 1
        renderer.scene.render()
        camera1 = renderer.scene.cameras[1]
        mask = camera1.img.to_numpy()
        # mask[:, :, 2] = mask[:, :, 0]
        # mask[:, :, 1] = mask[:, :, 0]
        ti.imwrite(mask, os.path.join(mask_save_path, '{}.png'.format(angle)))

    # for i in range(ran_mask_num+1):
    #     del scene.models[i]
    # for i in range(ran_mask_num):
    #     del mask_objs[i]

if __name__ == '__main__':
    res = (1024, 1024)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--texture_root", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--ran_mask_num", type=int, default=0)
    args = parser.parse_args()
    data_root = args.data_root
    texture_root = args.texture_root
    save_path = args.save_path
    ran_mask_num = args.ran_mask_num
    it = 0
    render_num = 0
    renderer = StaticRenderer()
    for data_id in tqdm(os.listdir(data_root), desc='data_id'):
        render_mv_random_mask(renderer, data_root, texture_root, data_id, save_path, res, False, dis_scale=2, ran_mask_num=ran_mask_num)
        render_num += 1
        if render_num > 50:
            break
