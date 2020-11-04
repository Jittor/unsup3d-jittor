import math
import jrender as jr
import numpy as np
import jittor as jt
from .utils import *

EPS = 1e-7

class Renderer():
    def __init__(self, cfgs):
        self.image_size = cfgs.get('image_size', 64)
        self.min_depth = cfgs.get('min_depth', 0.9)
        self.max_depth = cfgs.get('max_depth', 1.1)
        self.rot_center_depth = cfgs.get('rot_center_depth', (self.min_depth+self.max_depth)/2)
        self.fov = cfgs.get('fov', 10)
        self.tex_cube_size = cfgs.get('tex_cube_size', 2)
        self.renderer_min_depth = cfgs.get('renderer_min_depth', 0.1)
        self.renderer_max_depth = cfgs.get('renderer_max_depth', 10.)

        #### camera intrinsics
        #             (u)   (x)
        #    d * K^-1 (v) = (y)
        #             (1)   (z)

        ## renderer for visualization
        R = [[[1.,0.,0.],
              [0.,1.,0.],
              [0.,0.,1.]]]
        R = jt.array(R).float32()
        t = jt.zeros((1,3)).float32()
        fx = (self.image_size-1)/2/(math.tan(self.fov/2 *math.pi/180))
        fy = (self.image_size-1)/2/(math.tan(self.fov/2 *math.pi/180))
        cx = (self.image_size-1)/2
        cy = (self.image_size-1)/2
        K = [[fx, 0., cx],
             [0., fy, cy],
             [0., 0., 1.]]
        self.inv_K = jt.array(np.linalg.inv(np.array(K))).unsqueeze(0).float32()
        K = jt.array(K).float32()
        self.K = K.unsqueeze(0)
        self.renderer = jr.Renderer(camera_mode='projection',
                                    light_intensity_ambient=1.0,
                                    light_intensity_directionals=0.,
                                    K=self.K, R=R, t=t,
                                    near=self.renderer_min_depth, far=self.renderer_max_depth,
                                    image_size=self.image_size, orig_size=self.image_size,
                                    fill_back=True,
                                    background_color=[1,1,1],
                                    dr_type='n3mr')

    def set_transform_matrices(self, view):
        self.rot_mat, self.trans_xyz = get_transform_matrices(view)

    def rotate_pts(self, pts, rot_mat):
        centroid = jt.array([0.,0.,self.rot_center_depth]).view(1,1,3).float32()
        pts = pts - centroid  # move to centroid
        pts = pts.matmul(rot_mat.transpose((0,2,1)))  # rotate
        pts = pts + centroid  # move back
        return pts

    def translate_pts(self, pts, trans_xyz):
        return pts + trans_xyz

    def depth_to_3d_grid(self, depth):
        b, h, w = depth.shape
        grid_2d = get_grid(b, h, w, normalize=False)  # Nxhxwx2
        depth = depth.unsqueeze(-1)
        grid_3d = jt.contrib.concat((grid_2d, jt.ones(depth.shape)), dim=3)
        grid_3d = grid_3d.matmul(self.inv_K.transpose((0,2,1))[0]) * depth
        return grid_3d

    def grid_3d_to_2d(self, grid_3d):
        b, h, w, _ = grid_3d.shape
        # st()
        grid_2d = grid_3d / grid_3d[:,:,:,2:]
        grid_2d = grid_2d.matmul(self.K.transpose((0,2,1))[0])[:,:,:,:2]
        WH = jt.array([w-1, h-1]).view(1,1,1,2).float32()
        grid_2d = grid_2d / WH *2.-1.  # normalize to -1~1
        return grid_2d

    def get_warped_3d_grid(self, depth):
        b, h, w = depth.shape
        grid_3d = self.depth_to_3d_grid(depth).reshape(b,-1,3)
        grid_3d = self.rotate_pts(grid_3d, self.rot_mat)
        grid_3d = self.translate_pts(grid_3d, self.trans_xyz)
        return grid_3d.reshape(b,h,w,3) # return 3d vertices

    def get_inv_warped_3d_grid(self, depth):
        b, h, w = depth.shape
        # st()
        grid_3d = self.depth_to_3d_grid(depth).reshape(b,-1,3)
        grid_3d = self.translate_pts(grid_3d, -self.trans_xyz)
        grid_3d = self.rotate_pts(grid_3d, self.rot_mat.transpose((0,2,1)))
        return grid_3d.reshape(b,h,w,3) # return 3d vertices

    def get_warped_2d_grid(self, depth):
        b, h, w = depth.shape
        grid_3d = self.get_warped_3d_grid(depth)
        grid_2d = self.grid_3d_to_2d(grid_3d)
        return grid_2d

    def get_inv_warped_2d_grid(self, depth):
        b, h, w = depth.shape
        grid_3d = self.get_inv_warped_3d_grid(depth)
        grid_2d = self.grid_3d_to_2d(grid_3d)
        return grid_2d

    def warp_canon_depth(self, canon_depth):
        b, h, w = canon_depth.shape
        grid_3d = self.get_warped_3d_grid(canon_depth).reshape(b,-1,3)
        faces = get_face_idx(b, h, w)
        warped_depth = self.renderer(grid_3d, faces, mode='depth')

        # allow some margin out of valid range
        margin = (self.max_depth - self.min_depth) /2
        warped_depth = warped_depth.clamp(self.min_depth-margin, self.max_depth+margin)
        return warped_depth

    def get_normal_from_depth(self, depth):
        b, h, w = depth.shape
        grid_3d = self.depth_to_3d_grid(depth)

        tu = grid_3d[:,1:-1,2:] - grid_3d[:,1:-1,:-2]
        tv = grid_3d[:,2:,1:-1] - grid_3d[:,:-2,1:-1]
        normal = tu.cross(tv, dim=3)

        zero = jt.array([0,0,1]).float32()
        normal = jt.contrib.concat([zero.repeat(b,h-2,1,1), normal, zero.repeat(b,h-2,1,1)], 2)
        normal = jt.contrib.concat([zero.repeat(b,1,w,1), normal, zero.repeat(b,1,w,1)], 1)
        normal = normal / (((normal**2).sum(3, True))**0.5 + EPS)
        return normal

    def render_yaw(self, im, depth, v_before=None, v_after=None, rotations=None, maxr=90, nsample=9, crop_mesh=None):
        b, c, h, w = im.shape
        grid_3d = self.depth_to_3d_grid(depth)

        if crop_mesh is not None:
            top, bottom, left, right = crop_mesh  # pixels from border to be cropped
            if top > 0:
                grid_3d[:,:top,:,1] = grid_3d[:,top:top+1,:,1].repeat(1,top,1)
                grid_3d[:,:top,:,2] = grid_3d[:,top:top+1,:,2].repeat(1,top,1)
            if bottom > 0:
                grid_3d[:,-bottom:,:,1] = grid_3d[:,-bottom-1:-bottom,:,1].repeat(1,bottom,1)
                grid_3d[:,-bottom:,:,2] = grid_3d[:,-bottom-1:-bottom,:,2].repeat(1,bottom,1)
            if left > 0:
                grid_3d[:,:,:left,0] = grid_3d[:,:,left:left+1,0].repeat(1,1,left)
                grid_3d[:,:,:left,2] = grid_3d[:,:,left:left+1,2].repeat(1,1,left)
            if right > 0:
                grid_3d[:,:,-right:,0] = grid_3d[:,:,-right-1:-right,0].repeat(1,1,right)
                grid_3d[:,:,-right:,2] = grid_3d[:,:,-right-1:-right,2].repeat(1,1,right)

        grid_3d = grid_3d.reshape(b,-1,3)
        im_trans = []

        # inverse warp
        if v_before is not None:
            rot_mat, trans_xyz = get_transform_matrices(v_before)
            grid_3d = self.translate_pts(grid_3d, -trans_xyz)
            grid_3d = self.rotate_pts(grid_3d, rot_mat.transpose((0,2,1)))

        if rotations is None:
            rotations = np.linspace(-math.pi/180*maxr, math.pi/180*maxr, nsample)
        for i, ri in enumerate(rotations):
            ri = jt.array([0, ri, 0]).view(1,3).float32()
            rot_mat_i, _ = get_transform_matrices(ri)
            grid_3d_i = self.rotate_pts(grid_3d, rot_mat_i.repeat(b,1,1))

            if v_after is not None:
                if len(v_after.shape) == 3:
                    v_after_i = v_after[i]
                else:
                    v_after_i = v_after
                rot_mat, trans_xyz = get_transform_matrices(v_after_i)
                grid_3d_i = self.rotate_pts(grid_3d_i, rot_mat)
                grid_3d_i = self.translate_pts(grid_3d_i, trans_xyz)

            faces = get_face_idx(b, h, w)
            textures = get_textures_from_im(im, tx_size=self.tex_cube_size)
            warped_images = self.renderer(grid_3d_i, faces, textures, mode='rgb').clamp(min_v=-1., max_v=1.)
            im_trans += [warped_images]
        return jt.stack(im_trans, 1)  # b x t x c x h x w