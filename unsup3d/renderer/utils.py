import jittor as jt
import numpy as np

def mm_normalize(x, min=0, max=1):
    x_min = x.min()
    x_max = x.max()
    x_range = x_max - x_min
    x_z = (x - x_min) / x_range
    x_out = x_z * (max - min) + min
    return x_out


def rand_range(size, min, max):
    return jt.init.gauss(size, "float32")*(max-min)+min


def rand_posneg_range(size, min, max):
    i = (jt.init.gauss(size, "float32") > 0.5)*2.-1.
    return i*rand_range(size, min, max)


def get_grid(b, H, W, normalize=True):
    if normalize:
        h_range = np.linspace(-1,1,H)
        w_range = np.linspace(-1,1,W)
    else:
        h_range = np.arange(0,H)
        w_range = np.arange(0,W)
    grid = jt.array(np.stack(np.meshgrid(h_range, w_range), -1)[:,:,::-1]).repeat(b,1,1,1).flip(3).float() # flip h,w to x,y
    return grid


def get_rotation_matrix(tx, ty, tz):
    m_x = jt.zeros((tx.shape[0], 3, 3))
    m_y = jt.zeros((tx.shape[0], 3, 3))
    m_z = jt.zeros((tx.shape[0], 3, 3))

    m_x[:, 1, 1], m_x[:, 1, 2] = tx.cos(), -tx.sin()
    m_x[:, 2, 1], m_x[:, 2, 2] = tx.sin(), tx.cos()
    m_x[:, 0, 0] = 1

    m_y[:, 0, 0], m_y[:, 0, 2] = ty.cos(), ty.sin()
    m_y[:, 2, 0], m_y[:, 2, 2] = -ty.sin(), ty.cos()
    m_y[:, 1, 1] = 1

    m_z[:, 0, 0], m_z[:, 0, 1] = tz.cos(), -tz.sin()
    m_z[:, 1, 0], m_z[:, 1, 1] = tz.sin(), tz.cos()
    m_z[:, 2, 2] = 1
    return jt.matmul(m_z, jt.matmul(m_y, m_x))


def get_transform_matrices(view):
    b = view.size(0)
    if view.size(1) == 6:
        rx = view[:,0]
        ry = view[:,1]
        rz = view[:,2]
        trans_xyz = view[:,3:].reshape(b,1,3)
    elif view.size(1) == 5:
        rx = view[:,0]
        ry = view[:,1]
        rz = view[:,2]
        delta_xy = view[:,3:].reshape(b,1,2)
        trans_xyz = jt.contrib.concat([delta_xy, jt.zeros((b,1,1))], 2)
    elif view.size(1) == 3:
        rx = view[:,0]
        ry = view[:,1]
        rz = view[:,2]
        trans_xyz = jt.zeros((b,1,3))
    rot_mat = get_rotation_matrix(rx, ry, rz)
    return rot_mat, trans_xyz


def get_face_idx(b, h, w):
    idx_map = np.arange(h*w).reshape(h,w)
    faces1 = jt.array(np.stack([idx_map[:h-1,:w-1], idx_map[1:,:w-1], idx_map[:h-1,1:]], -1)).reshape(-1,3)
    faces2 = jt.array(np.stack([idx_map[:h-1,1:], idx_map[1:,:w-1], idx_map[1:,1:]], -1)).reshape(-1,3)
    return jt.contrib.concat([faces1,faces2], 0).repeat(b,1,1).int()


def vcolor_to_texture_cube(vcolors):
    # input bxcxnx3
    b, c, n, f = vcolors.shape
    coeffs = jt.array(
        [[ 0.5,  0.5,  0.5],
         [ 0. ,  0. ,  1. ],
         [ 0. ,  1. ,  0. ],
         [-0.5,  0.5,  0.5],
         [ 1. ,  0. ,  0. ],
         [ 0.5, -0.5,  0.5],
         [ 0.5,  0.5, -0.5],
         [ 0. ,  0. ,  0. ]])
    return coeffs.matmul(vcolors.permute(0,2,3,1)).reshape(b,n,2,2,2,c)


def get_textures_from_im(im, tx_size=1):
    b, c, h, w = im.shape
    if tx_size == 1:
        textures = jt.contrib.concat([im[:,:,:h-1,:w-1].reshape(b,c,-1), im[:,:,1:,1:].reshape(b,c,-1)], 2)
        textures = textures.transpose(2,1).reshape(b,-1,1,1,1,c)
    elif tx_size == 2:
        textures1 = jt.stack([im[:,:,:h-1,:w-1], im[:,:,:h-1,1:], im[:,:,1:,:w-1]], -1).reshape(b,c,-1,3)
        textures2 = jt.stack([im[:,:,1:,:w-1], im[:,:,:h-1,1:], im[:,:,1:,1:]], -1).reshape(b,c,-1,3)
        textures = vcolor_to_texture_cube(jt.contrib.concat([textures1, textures2], 2)) # bxnx2x2x2xc
    else:
        raise NotImplementedError("Currently support texture size of 1 or 2 only.")
    return textures
