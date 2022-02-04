import numpy as np
import torch
from .geometry import batch_cross_3d

def max_min_norm(x, b_min, b_max):
    '''
    :param x [B, 3, N]
    :param b_min [3] list
    :param b_max [3] list
    '''
    b_min = torch.FloatTensor(b_min).to(x.device).reshape(1, 3, 1)
    b_max = torch.FloatTensor(b_max).to(x.device).reshape(1, 3, 1)
    return (x - b_min) / (b_max - b_min)

def obj_sample_surface_wo_tex(obj, count):
    with torch.no_grad():
        vert = torch.FloatTensor(obj['vi'][:, :3])
        faces = torch.LongTensor(obj['f']).permute(0, 2, 1)
        v_tri = torch.zeros((faces.shape[0], 3, 3))
        for i in range(3):
            v_tri[:, i, :] = vert[faces[:, 0, i], :]
        v_ori = v_tri[:, 0, :].clone()
        v_vec = v_tri[:, 1:, :].clone() - v_ori.unsqueeze(1)
        
        area = torch.zeros((faces.shape[0], 1))
        cross = batch_cross_3d(v_vec[:, 0, :], v_vec[:, 1, :])
        area = torch.abs(torch.sum(cross, dim=1).unsqueeze(1)).numpy()
        
        area_sum = np.sum(area)
        area_cum = np.cumsum(area)
        face_pick = np.random.random(count) * area_sum
        face_index = np.searchsorted(area_cum, face_pick)-1
        
        v_ori = v_ori[face_index, :]
        v_vec = v_vec[face_index, :]
        
        random_lengths = np.random.random((count, 2, 1))

        # points will be distributed on a quadrilateral if we use 2 0-1 samples
        # if the two scalar components sum less than 1.0 the point will be
        # inside the triangle, so we find vectors longer than 1.0 and
        # transform them to be inside the triangle
        random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
        random_lengths[random_test] -= 1.0
        random_lengths = torch.FloatTensor(np.abs(random_lengths))

        # multiply triangle edge vectors by the random lengths and sum
        sample_v_vec = (v_vec * random_lengths).sum(dim=1)
        
        # finally, offset by the origin to generate
        v_samples = sample_v_vec + v_ori
        
    return v_samples

def obj_sample_surface(obj, count):
    with torch.no_grad():
        vert = torch.FloatTensor(obj['vi'])
        tex = torch.FloatTensor(obj['vt'])
        faces = torch.LongTensor(obj['f']).permute(0, 2, 1)
        v_tri = torch.zeros((faces.shape[0], 3, 3))
        t_tri = torch.zeros((faces.shape[0], 3, 2))
        for i in range(3):
            v_tri[:, i, :] = vert[faces[:, 0, i], :]
            t_tri[:, i, :] = tex[faces[:, 1, i], :]
        v_ori = v_tri[:, 0, :].clone()
        t_ori = t_tri[:, 0, :].clone()
        v_vec = v_tri[:, 1:, :].clone() - v_ori.unsqueeze(1)
        t_vec = t_tri[:, 1:, :].clone() - t_ori.unsqueeze(1)

        area = torch.zeros((faces.shape[0], 1))
        cross = batch_cross_3d(v_vec[:, 0, :], v_vec[:, 1, :])
        area = torch.abs(torch.sum(cross, dim=1).unsqueeze(1)).numpy()
        normal = cross / torch.sum(torch.sqrt(cross**2), dim=1).unsqueeze(1)
        
        area_sum = np.sum(area)
        # cumulative area (len(mesh.faces))
        area_cum = np.cumsum(area)
        face_pick = np.random.random(count) * area_sum
        face_index = np.searchsorted(area_cum, face_pick)-1
        
        v_ori, t_ori = v_ori[face_index, :], t_ori[face_index, :]
        v_vec, t_vec = v_vec[face_index, :], t_vec[face_index, :]

        random_lengths = np.random.random((count, 2, 1))

        # points will be distributed on a quadrilateral if we use 2 0-1 samples
        # if the two scalar components sum less than 1.0 the point will be
        # inside the triangle, so we find vectors longer than 1.0 and
        # transform them to be inside the triangle
        random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
        random_lengths[random_test] -= 1.0
        random_lengths = torch.FloatTensor(np.abs(random_lengths))

        # multiply triangle edge vectors by the random lengths and sum
        sample_v_vec = (v_vec * random_lengths).sum(dim=1)
        sample_t_vec = (t_vec * random_lengths).sum(dim=1)

        # finally, offset by the origin to generate
        v_samples = sample_v_vec + v_ori
        t_samples = sample_t_vec + t_ori
        n_samples = normal[face_index, :]

    return v_samples, t_samples, n_samples

def save_samples_truncted_prob_obj(fname, points, prob, continue_write=False, pts_num=0):
    '''
    Save the visualization of sampling to a obj file.
    Red points represent positive predictions.
    Green points represent negative predictions.
    :param fname: File name to save
    :param points: [N, 3] array of points
    :param prob: [N, 1] array of predictions in the range [0~1]
    :return:
    '''
    r = (prob).reshape([-1, 1]) * 255
    g = (1-prob).reshape([-1, 1]) * 255
    b = np.zeros(r.shape)
    if pts_num == 0:
        pts_num = points.shape[0]

    to_save = np.concatenate([points, r, g, b], axis=-1)
    if continue_write:
        f = open(fname, mode='a')
    else:
        f = open(fname, mode='w')
        # f.write('ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n' % pts_num)
    for i in range(points.shape[0]):
        f.write("v %f %f %f %d %d %d\n" % (points[i, 0], points[i, 1], points[i, 2], r[i], g[i], b[i]))

def save_samples_truncted_prob(fname, points, prob):
    '''
    Save the visualization of sampling to a ply file.
    Red points represent positive predictions.
    Green points represent negative predictions.
    :param fname: File name to save
    :param points: [N, 3] array of points
    :param prob: [N, 1] array of predictions in the range [0~1]
    :return:
    '''
    r = (prob).reshape([-1, 1]) * 255
    g = (1-prob).reshape([-1, 1]) * 255
    b = np.zeros(r.shape)

    to_save = np.concatenate([points, r, g, b], axis=-1)
    return np.savetxt(fname,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                          points.shape[0])
                      )


def save_samples_rgb(fname, points, rgb):
    '''
    Save the visualization of sampling to a ply file.
    Red points represent positive predictions.
    Green points represent negative predictions.
    :param fname: File name to save
    :param points: [N, 3] array of points
    :param rgb: [N, 3] array of rgb values in the range [0~1]
    :return:
    '''
    to_save = np.concatenate([points, rgb * 255], axis=-1)
    return np.savetxt(fname,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                          points.shape[0])
                      )
