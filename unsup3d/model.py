import os
import math
import glob
import jittor as jt
from jittor import nn
from . import networks
from . import utils
from .renderer import Renderer

EPS = 1e-7
cnt = 0

def print_var(var):
    global cnt
    cnt += 1
    print(cnt, var.shape, var.min(), var.max())

class Unsup3D():
    def __init__(self, cfgs):
        self.model_name = cfgs.get('model_name', self.__class__.__name__)
        self.device = cfgs.get('device', 'cpu')
        self.image_size = cfgs.get('image_size', 64)
        self.min_depth = cfgs.get('min_depth', 0.9)
        self.max_depth = cfgs.get('max_depth', 1.1)
        self.border_depth = cfgs.get('border_depth', (0.7*self.max_depth + 0.3*self.min_depth))
        self.xyz_rotation_range = cfgs.get('xyz_rotation_range', 60)
        self.xy_translation_range = cfgs.get('xy_translation_range', 0.1)
        self.z_translation_range = cfgs.get('z_translation_range', 0.1)
        self.lam_perc = cfgs.get('lam_perc', 1)
        self.lam_flip = cfgs.get('lam_flip', 0.5)
        self.lam_flip_start_epoch = cfgs.get('lam_flip_start_epoch', 0)
        self.lr = cfgs.get('lr', 1e-4)
        self.load_gt_depth = cfgs.get('load_gt_depth', False)
        self.renderer = Renderer(cfgs)

        ## networks and optimizers
        self.netD = networks.EDDeconv(cin=3, cout=1, nf=64, zdim=256, activation=None)
        self.netA = networks.EDDeconv(cin=3, cout=3, nf=64, zdim=256)
        self.netL = networks.Encoder(cin=3, cout=4, nf=32)
        self.netV = networks.Encoder(cin=3, cout=6, nf=32)
        self.netC = networks.ConfNet(cin=3, cout=2, nf=64, zdim=128)

        self.network_names = [k for k in vars(self) if 'net' in k]

        self.make_optimizer = lambda model: nn.Adam(model.parameters(),
            lr=self.lr, betas=(0.9, 0.999))
        ## other parameters
        self.PerceptualLoss = networks.PerceptualLoss(requires_grad=False)
        self.other_param_names = ['PerceptualLoss']

        ## depth rescaler: -1~1 -> min_deph~max_deph
        self.depth_rescaler = lambda d : (1+d)/2 *self.max_depth + (1-d)/2 *self.min_depth

    def init_optimizers(self):
        self.optimizer_names = []
        for net_name in self.network_names:
            optimizer = self.make_optimizer(getattr(self, net_name))
            optim_name = net_name.replace('net','optimizer')
            setattr(self, optim_name, optimizer)
            self.optimizer_names += [optim_name]

    def set_train(self):
        for net_name in self.network_names:
            getattr(self, net_name).train()

    def set_eval(self):
        for net_name in self.network_names:
            getattr(self, net_name).eval()

    def photometric_loss(self, im1, im2, mask=None, conf_sigma=None):
        loss = (im1-im2).abs()
        if conf_sigma is not None:
            loss = loss *2**0.5 / (conf_sigma +EPS) + (conf_sigma +EPS).log()
        if mask is not None:
            mask = mask.broadcast(loss.shape)
            loss = (loss * mask).sum() / (mask.sum() + EPS)
        else:
            loss = loss.mean()
        return loss

    def backward(self):
        for optim_name in self.optimizer_names:
            getattr(self, optim_name).step(self.loss_total)

    def forward(self, input):
        """Feedforward once."""
        if self.load_gt_depth:
            input, depth_gt = input
        self.input_im = input *2.-1.
        b, c, h, w = self.input_im.shape

        self.canon_depth_raw = self.netD(self.input_im).squeeze(1)  # BxHxW
        self.canon_depth = self.canon_depth_raw - self.canon_depth_raw.view(b,-1).mean(1).view(b,1,1)
        self.canon_depth = self.canon_depth.tanh()
        self.canon_depth = self.depth_rescaler(self.canon_depth)

        ## clamp border depth
        depth_border = jt.zeros((1,h,w-4))
        depth_border = nn.ConstantPad2d((2,2,0,0), 1)(depth_border)
        self.canon_depth = self.canon_depth*(1-depth_border) + depth_border *self.border_depth
        self.canon_depth = jt.contrib.concat([self.canon_depth, self.canon_depth.flip(2)], 0)  # flip

        ## predict canonical albedo
        self.canon_albedo = self.netA(self.input_im)  # Bx3xHxW
        self.canon_albedo = jt.contrib.concat([self.canon_albedo, self.canon_albedo.flip(3)], 0)  # flip
        ## predict confidence map
        self.conf_sigma_l1, self.conf_sigma_percl = self.netC(self.input_im)  # Bx2xHxW

        ## predict lighting
        canon_light = self.netL(self.input_im).repeat(2,1)  # Bx4
        self.canon_light_a = canon_light[:,:1] /2+0.5  # ambience term
        self.canon_light_b = canon_light[:,1:2] /2+0.5  # diffuse term
        canon_light_dxy = canon_light[:,2:]
        self.canon_light_d = jt.contrib.concat([canon_light_dxy, jt.ones((b*2,1))], 1)
        self.canon_light_d = self.canon_light_d / ((self.canon_light_d**2).sum(1, True)).sqrt()  # diffuse light direction

        ## shading
        self.canon_normal = self.renderer.get_normal_from_depth(self.canon_depth)
        self.canon_diffuse_shading = (self.canon_normal * self.canon_light_d.view(-1,1,1,3)).sum(3).clamp(min_v=0,max_v=1e5).unsqueeze(1)
        canon_shading = self.canon_light_a.view(-1,1,1,1) + self.canon_light_b.view(-1,1,1,1)*self.canon_diffuse_shading
        self.canon_im = (self.canon_albedo/2+0.5) * canon_shading *2-1

        ## predict viewpoint transformation
        self.view = self.netV(self.input_im).repeat(2,1)
        self.view = jt.contrib.concat([
            self.view[:,:3] *math.pi/180 *self.xyz_rotation_range,
            self.view[:,3:5] *self.xy_translation_range,
            self.view[:,5:] *self.z_translation_range], 1)

        ## reconstruct input view
        self.renderer.set_transform_matrices(self.view)
        self.recon_depth = self.renderer.warp_canon_depth(self.canon_depth)
        self.recon_normal = self.renderer.get_normal_from_depth(self.recon_depth)
        grid_2d_from_canon = self.renderer.get_inv_warped_2d_grid(self.recon_depth)
        
        self.recon_im = nn.grid_sample(self.canon_im, grid_2d_from_canon, mode='bilinear')
        margin = (self.max_depth - self.min_depth) /2
        recon_im_mask = (self.recon_depth < self.max_depth+margin).float32()  # invalid border pixels have been clamped at max_depth+margin
        recon_im_mask_both = recon_im_mask[:b] * recon_im_mask[b:]  # both original and flip reconstruction
        recon_im_mask_both = recon_im_mask_both.repeat(2,1,1).unsqueeze(1)
        self.recon_im = self.recon_im * recon_im_mask_both

        ## render symmetry axis
        canon_sym_axis = jt.zeros((h, w))
        canon_sym_axis[:, w//2-1:w//2+1] = 1
        self.recon_sym_axis = nn.grid_sample(canon_sym_axis.repeat(b*2,1,1,1), grid_2d_from_canon, mode='bilinear')
        self.recon_sym_axis = self.recon_sym_axis * recon_im_mask_both
        green = jt.array([-1,1,-1]).view(1,3,1,1)
        self.input_im_symline = (0.5*self.recon_sym_axis) *green + (1-0.5*self.recon_sym_axis) *self.input_im.repeat(2,1,1,1)

        ## loss function
        self.loss_l1_im = self.photometric_loss(self.recon_im[:b], self.input_im, mask=recon_im_mask_both[:b], conf_sigma=self.conf_sigma_l1[:,:1])
        self.loss_l1_im_flip = self.photometric_loss(self.recon_im[b:], self.input_im, mask=recon_im_mask_both[b:], conf_sigma=self.conf_sigma_l1[:,1:])
        self.loss_perc_im = self.PerceptualLoss(self.recon_im[:b], self.input_im, mask=recon_im_mask_both[:b], conf_sigma=self.conf_sigma_percl[:,:1])
        self.loss_perc_im_flip = self.PerceptualLoss(self.recon_im[b:], self.input_im, mask=recon_im_mask_both[b:], conf_sigma=self.conf_sigma_percl[:,1:])
        lam_flip = 1 if self.trainer.current_epoch < self.lam_flip_start_epoch else self.lam_flip
        self.loss_total = self.loss_l1_im + lam_flip*self.loss_l1_im_flip + self.lam_perc*(self.loss_perc_im + lam_flip*self.loss_perc_im_flip)

        self.loss_total.sync()
        metrics = {'loss': self.loss_total}

        ## compute accuracy if gt depth is available
        if self.load_gt_depth:
            self.depth_gt = depth_gt[:,0,:,:]
            self.depth_gt = (1-self.depth_gt)*2-1
            self.depth_gt = self.depth_rescaler(self.depth_gt)
            self.normal_gt = self.renderer.get_normal_from_depth(self.depth_gt)

            # mask out background
            mask_gt = (self.depth_gt<self.depth_gt.max()).float()
            mask_gt = (nn.Pool(3,1,1,op="mean")(mask_gt.unsqueeze(1)).squeeze(1) > 0.99).float()  # erode by 1 pixel
            mask_pred = (nn.Pool(3,1,1,op="mean")(recon_im_mask[:b].unsqueeze(1)).squeeze(1) > 0.99).float()  # erode by 1 pixel
            mask = mask_gt * mask_pred
            self.acc_mae_masked = ((self.recon_depth[:b] - self.depth_gt[:b]).abs() *mask).view(b,-1).sum(1) / mask.view(b,-1).sum(1)
            self.acc_mse_masked = (((self.recon_depth[:b] - self.depth_gt[:b])**2) *mask).view(b,-1).sum(1) / mask.view(b,-1).sum(1)
            self.sie_map_masked = utils.compute_sc_inv_err(self.recon_depth[:b].log(), self.depth_gt[:b].log(), mask=mask)
            self.acc_sie_masked = (self.sie_map_masked.view(b,-1).sum(1) / mask.view(b,-1).sum(1)).sqrt()
            self.norm_err_map_masked = utils.compute_angular_distance(self.recon_normal[:b], self.normal_gt[:b], mask=mask)
            self.acc_normal_masked = self.norm_err_map_masked.view(b,-1).sum(1) / mask.view(b,-1).sum(1)
            metrics['SIE_masked'] = self.acc_sie_masked.mean()
            metrics['NorErr_masked'] = self.acc_normal_masked.mean()

        return metrics

    def visualize(self, logger, total_iter, max_bs=25):
        b, c, h, w = self.input_im.shape
        b0 = min(max_bs, b)

        with jt.no_grad():
            ## render rotations
            v0 = jt.array([-0.1*math.pi/180*60,0,0,0,0,0]).repeat(b0,1)
            canon_im_rotate = self.renderer.render_yaw(self.canon_im[:b0], self.canon_depth[:b0], v_before=v0, maxr=90).detach() /2.+0.5  # (B,T,C,H,W)
            canon_normal_rotate = self.renderer.render_yaw(self.canon_normal[:b0].permute(0,3,1,2), self.canon_depth[:b0], v_before=v0, maxr=90).detach() /2.+0.5  # (B,T,C,H,W)

        input_im = self.input_im[:b0].detach().numpy() /2+0.5
        input_im_symline = self.input_im_symline[:b0].detach() /2.+0.5
        canon_albedo = self.canon_albedo[:b0].detach() /2.+0.5
        canon_im = self.canon_im[:b0].detach() /2.+0.5
        recon_im = self.recon_im[:b0].detach() /2.+0.5
        recon_im_flip = self.recon_im[b:b+b0].detach() /2.+0.5
        canon_depth_raw_hist = self.canon_depth_raw.detach().unsqueeze(1)
        canon_depth_raw = self.canon_depth_raw[:b0].detach().unsqueeze(1) /2.+0.5
        canon_depth = ((self.canon_depth[:b0] -self.min_depth)/(self.max_depth-self.min_depth)).detach().unsqueeze(1)
        recon_depth = ((self.recon_depth[:b0] -self.min_depth)/(self.max_depth-self.min_depth)).detach().unsqueeze(1)
        canon_diffuse_shading = self.canon_diffuse_shading[:b0].detach()
        canon_normal = self.canon_normal.permute(0,3,1,2)[:b0].detach() /2+0.5
        recon_normal = self.recon_normal.permute(0,3,1,2)[:b0].detach() /2+0.5
        conf_map_l1 = 1/(1+self.conf_sigma_l1[:b0,:1].detach()+EPS)
        conf_map_l1_flip = 1/(1+self.conf_sigma_l1[:b0,1:].detach()+EPS)
        conf_map_percl = 1/(1+self.conf_sigma_percl[:b0,:1].detach()+EPS)
        conf_map_percl_flip = 1/(1+self.conf_sigma_percl[:b0,1:].detach()+EPS)

        canon_im_rotate_grid = [jt.make_grid(img, nrow=int(math.ceil(math.sqrt(b0)))) for img in jt.unbind(canon_im_rotate, 1)]  # [(C,H,W)]*T
        canon_im_rotate_grid = jt.stack(canon_im_rotate_grid, 0).unsqueeze(0)  # (1,T,C,H,W)
        canon_normal_rotate_grid = [jt.make_grid(img, nrow=int(math.ceil(math.sqrt(b0)))) for img in jt.unbind(canon_normal_rotate, 1)]  # [(C,H,W)]*T
        canon_normal_rotate_grid = jt.stack(canon_normal_rotate_grid, 0).unsqueeze(0)  # (1,T,C,H,W)

        ## write summary
        logger.add_scalar('Loss/loss_total', self.loss_total.numpy(), total_iter)
        logger.add_scalar('Loss/loss_l1_im', self.loss_l1_im.numpy(), total_iter)
        logger.add_scalar('Loss/loss_l1_im_flip', self.loss_l1_im_flip.numpy(), total_iter)
        logger.add_scalar('Loss/loss_perc_im', self.loss_perc_im.numpy(), total_iter)
        logger.add_scalar('Loss/loss_perc_im_flip', self.loss_perc_im_flip.numpy(), total_iter)

        logger.add_histogram('Depth/canon_depth_raw_hist', canon_depth_raw_hist.numpy(), total_iter)
        vlist = ['view_rx', 'view_ry', 'view_rz', 'view_tx', 'view_ty', 'view_tz']
        for i in range(self.view.shape[1]):
            logger.add_histogram('View/'+vlist[i], self.view[:,i].numpy(), total_iter)
        logger.add_histogram('Light/canon_light_a', self.canon_light_a.numpy(), total_iter)
        logger.add_histogram('Light/canon_light_b', self.canon_light_b.numpy(), total_iter)
        llist = ['canon_light_dx', 'canon_light_dy', 'canon_light_dz']
        for i in range(self.canon_light_d.shape[1]):
            logger.add_histogram('Light/'+llist[i], self.canon_light_d[:,i].numpy(), total_iter)

        def log_grid_image(label, im, nrow=int(math.ceil(math.sqrt(b0))), iter=total_iter):
            im_grid = jt.make_grid(im, nrow=nrow)
            logger.add_image(label, im_grid.numpy(), iter)

        log_grid_image('Image/input_image_symline', input_im_symline)
        log_grid_image('Image/canonical_albedo', canon_albedo)
        log_grid_image('Image/canonical_image', canon_im)
        log_grid_image('Image/recon_image', recon_im)
        log_grid_image('Image/recon_image_flip', recon_im_flip)
        log_grid_image('Image/recon_side', canon_im_rotate[:,0,:,:,:])

        log_grid_image('Depth/canonical_depth_raw', canon_depth_raw)
        log_grid_image('Depth/canonical_depth', canon_depth)
        log_grid_image('Depth/recon_depth', recon_depth)
        log_grid_image('Depth/canonical_diffuse_shading', canon_diffuse_shading)
        log_grid_image('Depth/canonical_normal', canon_normal)
        log_grid_image('Depth/recon_normal', recon_normal)

        logger.add_histogram('Image/canonical_albedo_hist', canon_albedo.numpy(), total_iter)
        logger.add_histogram('Image/canonical_diffuse_shading_hist', canon_diffuse_shading.numpy(), total_iter)

        log_grid_image('Conf/conf_map_l1', conf_map_l1)
        logger.add_histogram('Conf/conf_sigma_l1_hist', self.conf_sigma_l1[:,:1].numpy(), total_iter)
        log_grid_image('Conf/conf_map_l1_flip', conf_map_l1_flip)
        logger.add_histogram('Conf/conf_sigma_l1_flip_hist', self.conf_sigma_l1[:,1:].numpy(), total_iter)
        log_grid_image('Conf/conf_map_percl', conf_map_percl)
        logger.add_histogram('Conf/conf_sigma_percl_hist', self.conf_sigma_percl[:,:1].numpy(), total_iter)
        log_grid_image('Conf/conf_map_percl_flip', conf_map_percl_flip)
        logger.add_histogram('Conf/conf_sigma_percl_flip_hist', self.conf_sigma_percl[:,1:].numpy(), total_iter)

        logger.add_video('Image_rotate/recon_rotate', canon_im_rotate_grid.numpy(), total_iter, fps=4)
        logger.add_video('Image_rotate/canon_normal_rotate', canon_normal_rotate_grid.numpy(), total_iter, fps=4)

        # visualize images and accuracy if gt is loaded
        if self.load_gt_depth:
            depth_gt = ((self.depth_gt[:b0] - self.min_depth)/(self.max_depth-self.min_depth)).detach().unsqueeze(1)
            normal_gt = self.normal_gt.permute(0,3,1,2)[:b0].detach() /2+0.5
            sie_map_masked = self.sie_map_masked[:b0].detach().unsqueeze(1) *1000
            norm_err_map_masked = self.norm_err_map_masked[:b0].detach().unsqueeze(1) /100

            logger.add_scalar('Acc_masked/MAE_masked', self.acc_mae_masked.mean().numpy(), total_iter)
            logger.add_scalar('Acc_masked/MSE_masked', self.acc_mse_masked.mean().numpy(), total_iter)
            logger.add_scalar('Acc_masked/SIE_masked', self.acc_sie_masked.mean().numpy(), total_iter)
            logger.add_scalar('Acc_masked/NorErr_masked', self.acc_normal_masked.mean().numpy(), total_iter)

            log_grid_image('Depth_gt/depth_gt', depth_gt)
            log_grid_image('Depth_gt/normal_gt', normal_gt)
            log_grid_image('Depth_gt/sie_map_masked', sie_map_masked)
            log_grid_image('Depth_gt/norm_err_map_masked', norm_err_map_masked)

    def save_results(self, save_dir):
        b, c, h, w = self.input_im.shape

        with jt.no_grad():
            v0 = jt.array([-0.1*math.pi/180*60,0,0,0,0,0]).repeat(b,1)
            canon_im_rotate = self.renderer.render_yaw(self.canon_im[:b], self.canon_depth[:b], v_before=v0, maxr=90, nsample=15)  # (B,T,C,H,W)
            canon_im_rotate = canon_im_rotate.clamp(-1,1) /2+0.5
            canon_normal_rotate = self.renderer.render_yaw(self.canon_normal[:b].permute(0,3,1,2), self.canon_depth[:b], v_before=v0, maxr=90, nsample=15)  # (B,T,C,H,W)
            canon_normal_rotate = canon_normal_rotate.clamp(-1,1) /2+0.5

        input_im = self.input_im[:b].numpy() /2+0.5
        input_im_symline = self.input_im_symline.numpy() /2.+0.5
        canon_albedo = self.canon_albedo[:b].numpy() /2+0.5
        canon_im = self.canon_im[:b].clamp(-1,1).numpy() /2+0.5
        recon_im = self.recon_im[:b].clamp(-1,1).numpy() /2+0.5
        recon_im_flip = self.recon_im[b:].clamp(-1,1).numpy() /2+0.5
        canon_depth = ((self.canon_depth[:b] -self.min_depth)/(self.max_depth-self.min_depth)).clamp(0,1).unsqueeze(1).numpy()
        recon_depth = ((self.recon_depth[:b] -self.min_depth)/(self.max_depth-self.min_depth)).clamp(0,1).unsqueeze(1).numpy()
        canon_diffuse_shading = self.canon_diffuse_shading[:b].numpy()
        canon_normal = self.canon_normal[:b].permute(0,3,1,2).numpy() /2+0.5
        recon_normal = self.recon_normal[:b].permute(0,3,1,2).numpy() /2+0.5
        conf_map_l1 = 1/(1+self.conf_sigma_l1[:b,:1].numpy()+EPS)
        conf_map_l1_flip = 1/(1+self.conf_sigma_l1[:b,1:].numpy()+EPS)
        conf_map_percl = 1/(1+self.conf_sigma_percl[:b,:1].numpy()+EPS)
        conf_map_percl_flip = 1/(1+self.conf_sigma_percl[:b,1:].numpy()+EPS)
        canon_light = jt.contrib.concat([self.canon_light_a, self.canon_light_b, self.canon_light_d], 1)[:b].numpy()
        view = self.view[:b].numpy()

        canon_im_rotate_grid = [jt.make_grid(img, nrow=int(math.ceil(math.sqrt(b)))) for img in jt.unbind(canon_im_rotate,1)]  # [(C,H,W)]*T

        canon_im_rotate_grid = jt.stack(canon_im_rotate_grid, 0).unsqueeze(0).numpy()  # (1,T,C,H,W)
        canon_normal_rotate_grid = [jt.make_grid(img, nrow=int(math.ceil(math.sqrt(b)))) for img in jt.unbind(canon_normal_rotate,1)]  # [(C,H,W)]*T
        canon_normal_rotate_grid = jt.stack(canon_normal_rotate_grid, 0).unsqueeze(0).numpy()  # (1,T,C,H,W)

        sep_folder = True
        utils.save_images(save_dir, input_im, suffix='input_image', sep_folder=sep_folder)
        utils.save_images(save_dir, input_im_symline, suffix='input_image_symline', sep_folder=sep_folder)
        utils.save_images(save_dir, canon_albedo, suffix='canonical_albedo', sep_folder=sep_folder)
        utils.save_images(save_dir, canon_im, suffix='canonical_image', sep_folder=sep_folder)
        utils.save_images(save_dir, recon_im, suffix='recon_image', sep_folder=sep_folder)
        utils.save_images(save_dir, recon_im_flip, suffix='recon_image_flip', sep_folder=sep_folder)
        utils.save_images(save_dir, canon_depth, suffix='canonical_depth', sep_folder=sep_folder)
        utils.save_images(save_dir, recon_depth, suffix='recon_depth', sep_folder=sep_folder)
        utils.save_images(save_dir, canon_diffuse_shading, suffix='canonical_diffuse_shading', sep_folder=sep_folder)
        utils.save_images(save_dir, canon_normal, suffix='canonical_normal', sep_folder=sep_folder)
        utils.save_images(save_dir, recon_normal, suffix='recon_normal', sep_folder=sep_folder)
        utils.save_images(save_dir, conf_map_l1, suffix='conf_map_l1', sep_folder=sep_folder)
        utils.save_images(save_dir, conf_map_l1_flip, suffix='conf_map_l1_flip', sep_folder=sep_folder)
        utils.save_images(save_dir, conf_map_percl, suffix='conf_map_percl', sep_folder=sep_folder)
        utils.save_images(save_dir, conf_map_percl_flip, suffix='conf_map_percl_flip', sep_folder=sep_folder)
        utils.save_txt(save_dir, canon_light, suffix='canonical_light', sep_folder=sep_folder)
        utils.save_txt(save_dir, view, suffix='viewpoint', sep_folder=sep_folder)

        utils.save_videos(save_dir, canon_im_rotate_grid, suffix='image_video', sep_folder=sep_folder, cycle=True)
        utils.save_videos(save_dir, canon_normal_rotate_grid, suffix='normal_video', sep_folder=sep_folder, cycle=True)

        # save scores if gt is loaded
        if self.load_gt_depth:
            depth_gt = ((self.depth_gt[:b] -self.min_depth)/(self.max_depth-self.min_depth)).clamp(0,1).detach().unsqueeze(1).numpy()
            normal_gt = self.normal_gt[:b].permute(0,3,1,2).detach().numpy() /2+0.5
            utils.save_images(save_dir, depth_gt, suffix='depth_gt', sep_folder=sep_folder)
            utils.save_images(save_dir, normal_gt, suffix='normal_gt', sep_folder=sep_folder)

            all_scores = jt.stack([
                self.acc_mae_masked.detach(),
                self.acc_mse_masked.detach(),
                self.acc_sie_masked.detach(),
                self.acc_normal_masked.detach()], 1)
            if not hasattr(self, 'all_scores'):
                self.all_scores = all_scores
            else:
                self.all_scores = jt.contrib.concat([self.all_scores, all_scores], 0)

    def save_scores(self, path):
        # save scores if gt is loaded
        if self.load_gt_depth:
            header = 'MAE_masked, \
                      MSE_masked, \
                      SIE_masked, \
                      NorErr_masked'
            mean = self.all_scores.numpy().mean(0)
            std = self.all_scores.numpy().std(0)
            header = header + '\nMean: ' + ',\t'.join(['%.8f'%x for x in mean])
            header = header + '\nStd: ' + ',\t'.join(['%.8f'%x for x in std])
            utils.save_scores(path, self.all_scores.numpy(), header=header)
