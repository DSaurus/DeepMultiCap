from .geometry import *

def calc_smpl_normal(smpl):
    vi = smpl['vi']
    vn = np.zeros((vi.shape[0], 3))
    for f in smpl['f']:
        a, b, c = vi[f[0, 0]], vi[f[1, 0]], vi[f[2, 0]]
        n = cross_3d(c-a, b-a)
        vn[f[0, 0]] += n
        vn[f[1, 0]] += n
        vn[f[2, 0]] += n
    vn = vn / np.sqrt(np.sum(vn**2, axis=1)).reshape((-1, 1))
    return vn

def smpl_normalize(smpl, norm_faces, flip_normal=False, init_rot=None):
    b_min = np.zeros(3)
    b_max = np.zeros(3)
    scale = 1.0
    center = np.zeros(3)
    
    vi = smpl['vi'].copy()[:, :3]
    b0 = np.min(vi, axis=0)
    b1 = np.max(vi, axis=0)
    center = (b0 + b1) / 2
    scale = np.min(1.0 / (b1 - b0)) * 0.9
    b_min = center - 0.5 / scale
    b_max = center + 0.5 / scale

    normal = np.zeros((3))
    for f in norm_faces:
        a, b, c = vi[f[0]][0], vi[f[1]][0], vi[f[2]][0]
        normal += cross_3d(c - a, b - a)
    if flip_normal:
        normal = -normal
    
    x, z = normal[0], normal[2]
    theta = math.acos(z / math.sqrt(z*z +x*x))
    if x < 0:
        theta = 2*math.acos(-1) - theta
    rot = rotationY(-theta)

    vi -= center
    vi *= scale
    if init_rot is not None:
        x, y, z = init_rot
        rot = np.array(rotationX(x / 180 * math.acos(-1))) @ rot
        rot = np.array(rotationY(y / 180 * math.acos(-1))) @ rot
        rot = np.array(rotationZ(z / 180 * math.acos(-1))) @ rot

    vi = (rot @ vi.T).T

    return {
        'b_min' : b_min,
        'b_max' : b_max,
        'scale' : scale,
        'center' : center,
        'direction' : normal,
        'smpl' : vi
    }