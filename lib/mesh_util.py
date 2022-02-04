from skimage import measure
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
import trimesh

from .sdf import create_grid, eval_grid_octree, eval_grid
from .net_util import reshape_sample_tensor
from .geometry import index


def gen_mesh_dmc(opt, net, cuda, data, save_path, use_octree=True, threshold=0.5):
    image_tensor = data['image'].to(device=cuda)
    calib_tensor = data['calib'].to(device=cuda)
    extrinsic = data['extrinsic'].to(device=cuda).unsqueeze(0)
    vox_tensor = data['vox'].to(device=cuda).unsqueeze(0)
    smpl_normal = data['smpl_normal'].to(device=cuda).unsqueeze(0)
    save_smpl_normal = smpl_normal.clone()
    normal_tensor = data['normal'].to(device=cuda)
    scale, center = data['scale'].to(device=cuda).unsqueeze(0), data['center'].to(device=cuda).unsqueeze(0)
    mask, ero_mask = data['mask'].to(device=cuda).unsqueeze(0), data['ero_mask'].to(device=cuda).unsqueeze(0)

    net.mask_init(mask, ero_mask)
    net.norm_init(scale, center)
    net.smpl_init(smpl_normal)
    
    net.filter2d(torch.cat([image_tensor.unsqueeze(0), smpl_normal], dim=2))
    if opt.fine_part:
        if normal_tensor.shape[2] == 1024:
            print('1024')
            smpl_normal = torch.nn.Upsample(size=[1024, 1024], mode='bilinear')(smpl_normal.squeeze(0)).unsqueeze(0)
        net.filter_normal(torch.cat([normal_tensor.unsqueeze(0), smpl_normal], dim=2))
    
    net.filter3d(vox_tensor)

    b_min = data['b_min']
    b_max = data['b_max']

    save_img_path = save_path[:-4] + '.png'
    save_img_list = []
    for v in range(image_tensor.shape[0]):
        save_img = (np.transpose(image_tensor[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
        save_smpl = (np.transpose(save_smpl_normal[0][v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
        save_img_list.append(save_img / 2 + save_smpl / 2)
    for v in range(normal_tensor.shape[0]):
        save_nm = normal_tensor[v]
        save_nm = F.interpolate(save_nm.unsqueeze(0), size=[512, 512], mode='bilinear')[0]
        save_nm = (np.transpose(save_nm.detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
        save_img_list.append(save_nm)
        # save_nm = smpl_normal[0:, v]
        # save_nm = F.interpolate(save_nm, size=[512, 512], mode='bilinear')[0]
        # save_nm = (np.transpose(save_nm.detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
        # save_img_list.append(save_nm)
    save_img = np.concatenate(save_img_list, axis=1)
    Image.fromarray(np.uint8(save_img[:,:,::-1])).save(save_img_path)

    try:
        verts, faces, _, _ = reconstruction_3d(
            net, cuda, calib_tensor.unsqueeze(0), extrinsic, opt.resolution, b_min, b_max, use_octree=use_octree, threshold=threshold)
        verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float()
        xyz_tensor = net.projection(verts_tensor, calib_tensor[:1])
        uv = xyz_tensor[:, :2, :]
        color = index(image_tensor[:1], uv).detach().cpu().numpy()[0].T
        color = color * 0.5 + 0.5
        save_obj_mesh_with_color(save_path, verts, faces, color)
    except Exception as e:
        print(e)


def reconstruction_3d(net, cuda, calib_tensor, extrinsic, 
                   resolution, b_min, b_max,
                   net_3d=False, use_octree=False, num_samples=30000, threshold=0.5, transform=None):
    '''
    Reconstruct meshes from sdf predicted by the network.
    :param net: a BasePixImpNet object. call image filter beforehead.
    :param cuda: cuda device
    :param calib_tensor: calibration tensor
    :param resolution: resolution of the grid cell
    :param b_min: bounding box corner [x_min, y_min, z_min]
    :param b_max: bounding box corner [x_max, y_max, z_max]
    :param use_octree: whether to use octree acceleration
    :param num_samples: how many points to query each gpu iteration
    :return: marching cubes results.
    '''
    # First we create a grid by resolution
    # and transforming matrix for grid coordinates to real world xyz
    #print(b_min, b_max)
    coords, mat = create_grid(resolution, resolution, resolution,
                              b_min, b_max, transform=transform)
    # print(coords.shape, mat.shape)
    # Then we define the lambda function for cell evaluation
    def eval_func(points):
        # print(points.shape)
        points = np.expand_dims(points, axis=0)
        samples = torch.from_numpy(points).to(device=cuda).float()
        net.query(samples, calib_tensor, extrinsic)
        pred = net.get_preds()[0][0]
        return pred.detach().cpu().numpy()

    # Then we evaluate the grid
    if use_octree:
        sdf = eval_grid_octree(coords, eval_func, num_samples=num_samples)
    else:
        sdf = eval_grid(coords, eval_func, num_samples=num_samples)

    # Finally we do marching cubes
    #try:
    verts, faces, normals, values = measure.marching_cubes_lewiner(sdf, threshold)
    # transform verts into world coordinate system
    verts = np.matmul(mat[:3, :3], verts.T) + mat[:3, 3:4]
    verts = verts.T
    return verts, faces, normals, values
    #except:
    #    print('error cannot marching cubes')
    #    return -1

def save_obj_mesh(mesh_path, verts, faces):
    file = open(mesh_path, 'w')

    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_obj_mesh_with_color(mesh_path, verts, faces, colors, reverse=False):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
    for f in faces:
        f_plus = f + 1
        if reverse:
            file.write('f %d %d %d\n' % (f_plus[0], f_plus[1], f_plus[2]))
        else:
            file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_obj_mesh_with_uv(mesh_path, verts, faces, uvs):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        vt = uvs[idx]
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
        file.write('vt %.4f %.4f\n' % (vt[0], vt[1]))

    for f in faces:
        f_plus = f + 1
        file.write('f %d/%d %d/%d %d/%d\n' % (f_plus[0], f_plus[0],
                                              f_plus[2], f_plus[2],
                                              f_plus[1], f_plus[1]))
    file.close()



def _append(faces, indices):
    if len(indices) == 4:
        faces.append([indices[0], indices[1], indices[2]])
        faces.append([indices[2], indices[3], indices[0]])
    elif len(indices) == 3:
        faces.append(indices)
    else:
        assert False, len(indices)


def readobj(path, scale=1):
    vi = []
    vt = []
    vn = []
    faces = []

    with open(path, 'r') as myfile:
        lines = myfile.readlines()

    # cache vertices
    for line in lines:
        try:
            type, fields = line.split(maxsplit=1)
            fields = [float(_) for _ in fields.split()]
        except ValueError:
            continue

        if type == 'v':
            vi.append(fields)
        elif type == 'vt':
            vt.append(fields)
        elif type == 'vn':
            vn.append(fields)

    # cache faces
    for line in lines:
        try:
            type, fields = line.split(maxsplit=1)
            fields = fields.split()
        except ValueError:
            continue

        # line looks like 'f 5/1/1 1/2/1 4/3/1'
        # or 'f 314/380/494 382/400/494 388/550/494 506/551/494' for quads
        if type != 'f':
            continue

        # a field should look like '5/1/1'
        # for vertex/vertex UV coords/vertex Normal  (indexes number in the list)
        # the index in 'f 5/1/1 1/2/1 4/3/1' STARTS AT 1 !!!
        indices = [[int(_) - 1 if _ != '' else 0 for _ in field.split('/')] for field in fields]

        if len(indices) == 4:
            faces.append([indices[0], indices[1], indices[2]])
            faces.append([indices[2], indices[3], indices[0]])
        elif len(indices) == 3:
            faces.append(indices)
        else:
            assert False, len(indices)

    ret = {}
    ret['vi'] = None if len(vi) == 0 else np.array(vi).astype(np.float32) * scale
    ret['vt'] = None if len(vt) == 0 else np.array(vt).astype(np.float32)
    ret['vn'] = None if len(vn) == 0 else np.array(vn).astype(np.float32)
    ret['f'] = None if len(faces) == 0 else np.array(faces).astype(np.int32)
    return ret

