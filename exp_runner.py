import os
import time
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from icecream import ic
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.dataset import Dataset
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from models.renderer import NeuSRenderer

from matplotlib import pyplot as plt

from scipy.sparse import coo_matrix
from collections import defaultdict

import lib.workspace as ws

import sys
sys.path.append('custom_mc')
from _marching_cubes_lewiner import udf_mc_lewiner

# import pymeshlab as ml

class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = Dataset(self.conf['dataset'])
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        # self.learning_rate_sdf = self.conf.get_float('train.learning_rate_sdf')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)

        self.weight_anneal_end = self.conf.get_float('train.weight_anneal_end', default=0.)

        self.mcube_threshold = self.conf.get_float('train.mcube_threshold')

        self.up_sample_start = self.conf.get_int('train.up_sample_start')

        self.perm_num = self.conf.get_int('train.perm_num')

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.normal_weight = self.conf.get_float('train.normal_weight')

        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)

        # params_to_train += list(self.nerf_outside.parameters())
        # params_to_train += list(self.deviation_network.parameters())
        # params_to_train += list(self.color_network.parameters())

        # params_to_train_sdf = list(self.sdf_network.parameters())
        # self.optimizer = torch.optim.Adam([{'params':params_to_train}, {'params':params_to_train_sdf}], lr=self.learning_rate)
        
        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)


        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     **self.conf['model.neus_renderer'])
        self.n_samples = self.renderer.n_samples

        # Load checkpoint
        latest_model_name = None
        if is_continue and os.path.exists(os.path.join(self.base_exp_dir, 'checkpoints')):
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]):
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Found checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()

    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = []
        for _ in range(self.perm_num):
            image_perm.append(self.get_image_perm())

        for iter_i in tqdm(range(res_step)):
            self.patch_size = 1
            self.patch_num = int(self.batch_size/self.patch_size/self.patch_size)
            self.renderer.patch_num = self.patch_num
            self.renderer.patch_size = self.patch_size
            
            for i in range(self.perm_num):
                if i == 0:
                    data = self.dataset.gen_random_rays_at(image_perm[0][self.iter_step % len(image_perm)],
                                                            int(self.batch_size/self.perm_num))
                else:
                    data_p = self.dataset.gen_random_rays_at(image_perm[i][self.iter_step % len(image_perm)],
                                                                int(self.batch_size/self.perm_num))
                    data = torch.cat([data, data_p], dim=0)

            rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = torch.ones([1, 3])

            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)

            mask_sum = mask.sum() + 1e-5
            up_sample = False
            if self.iter_step >= self.up_sample_start:
                up_sample = True
            self.renderer.iter_step = self.iter_step

            render_out = self.renderer.render(rays_o, rays_d, near, far,
                                              background_rgb=background_rgb,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              weight_anneal_ratio=self.get_weight_anneal_ratio(),
                                              up_sample=up_sample,
                                              )

            color_fine = render_out['color_fine']
            s_val = render_out['s_val']
            cdf_fine = render_out['cdf_fine']
            gradient_error = render_out['gradient_error']
            weight_max = render_out['weight_max']
            weight_sum = render_out['weight_sum']
            normal_error = render_out['normal_error']

            # Loss
            color_error = (color_fine - true_rgb) * mask
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
            psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

            eikonal_loss = gradient_error

            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

            normal_loss = normal_error

            loss = color_fine_loss
            loss += eikonal_loss * self.igr_weight
            loss += mask_loss * self.mask_weight
            loss += normal_loss * self.normal_weight

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1

            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
            self.writer.add_scalar('Loss/normal_loss', normal_loss, self.iter_step)

            self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
            self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0:
                self.validate_image(up_sample=True)

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh(world_space=True, resolution=256, threshold=self.mcube_threshold)

            self.update_learning_rate()

            if self.iter_step % self.perm_num == 0:
                image_perm = []
                for _ in range(self.perm_num):
                    image_perm.append(self.get_image_perm())

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def get_weight_anneal_ratio(self):
        if self.weight_anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.weight_anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor
        # self.optimizer.param_groups[0]['lr'] = self.learning_rate * learning_factor
        # self.optimizer.param_groups[1]['lr'] = self.learning_rate_sdf * learning_factor

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate_image(self, idx=-1, resolution_level=-1, up_sample=True):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb,
                                              up_sample=up_sample,
                                              )

            def feasible(key): return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           np.concatenate([img_fine[..., i],
                                           self.dataset.image_at(idx, resolution_level=resolution_level)]))
            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normals',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           normal_img[..., i])

    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

            del render_out

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        return img_fine

    def validate_mesh(self, world_space=False, resolution=64, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

        logging.info('End')

    def validate_point(self, num_steps=10, world_space=True, patch_size=10000, color=False):
        point_cloud, duration = self.renderer.generate_point_cloud(num_steps=num_steps, device=self.device, num_points=1000000, filter_val=0.001)

        print('num_steps', num_steps, 'duration', duration)

        if color:
            color_cpu = []
            while len(color_cpu) < len(point_cloud):
                print(len(color_cpu))
                min_idx = len(color_cpu)
                max_idx = np.min([len(point_cloud), min_idx+patch_size])
                color = self.get_vertices_color(vertices=torch.from_numpy(point_cloud[min_idx:max_idx, :].astype(np.float32)).to(self.device))
                if len(color_cpu)==0:
                    color_cpu = color
                else:
                    color_cpu = np.vstack((color_cpu, color))
        else:
            color_cpu = np.ones_like(point_cloud)

        if world_space:
            point_cloud = point_cloud * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        cloud = trimesh.points.PointCloud(vertices=point_cloud, colors=color_cpu[:, ::-1])
        cloud.export(os.path.join(self.base_exp_dir, 'meshes', 'pc{:0>8d}.ply'.format(self.iter_step)))

    def interpolate_view(self, img_idx_0, img_idx_1):
        images = []
        n_frames = 60
        for i in range(n_frames):
            print(i)
            images.append(self.render_novel_image(img_idx_0,
                                                  img_idx_1,
                                                  np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                          resolution_level=4))
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_dir = os.path.join(self.base_exp_dir, 'render')
        os.makedirs(video_dir, exist_ok=True)
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(os.path.join(video_dir,
                                             '{:0>8d}_{}_{}.mp4'.format(self.iter_step, img_idx_0, img_idx_1)),
                                fourcc, 30, (w, h))

        for image in images:
            writer.write(image)

        writer.release()

    def validate_mesh_udf_backups(self, world_space=False, resolution=64, color=False):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)
        vertices, triangles, vertices_color = validate_mesh_udf.main(bound_min, bound_max, self.sdf_network, self.color_network,  resolution, color)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        if color:
            mesh.visual.vertex_colors = vertices_color
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', 'mu{:0>8d}.ply'.format(self.iter_step)))

        logging.info('End')

    def validate_ray(self, idx=-1):
        if idx < 0:
            idx = self.dataset.n_images // 2

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        rays_o, rays_d = self.dataset.gen_rays_at(idx)
        H, W, _ = rays_o.shape
        ray_o = rays_o[H//2, W//2, :].reshape(1, 3)
        ray_d = rays_d[H//2, W//2, :].reshape(1, 3)

        near, far = self.dataset.near_far_from_sphere(ray_o, ray_d)
        background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None
        z_vals, sdf, weights = self.renderer.generate_ray(ray_o,
                                          ray_d,
                                          near,
                                          far,
                                          cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                          background_rgb=background_rgb)
        weights = weights/(z_vals[1:]-z_vals[:-1])

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_ray'), exist_ok=True)

        plt.plot(z_vals, sdf, marker='1')
        plt.plot((z_vals[1:] / 2 + z_vals[:-1] / 2), weights/weights.max())
        plt.savefig(os.path.join(self.base_exp_dir, 'validations_ray', '{:0>8d}_{}.png'.format(self.iter_step, idx)))
        plt.close()

    def get_vertices_color(self, vertices):
        pts = vertices
        dirs = torch.zeros_like(pts)
        sdf_nn_output = self.sdf_network(pts)
        sdf = sdf_nn_output[:, :1]
        feature_vector = sdf_nn_output[:, 1:]
        print(self.color_network.mode)
        
        if self.color_network.mode == 'second_order_udf':
            second_order_gradients, gradients = self.sdf_network.second_order_gradient(pts)
            second_order_gradients = second_order_gradients.squeeze()
            gradients = gradients.squeeze()
        else:
            second_order_gradients = None
            gradients = self.sdf_network.gradient(pts).squeeze()
        sampled_color = self.color_network(pts, gradients, dirs, feature_vector, sdf, second_order_gradients).reshape(-1, 3)
        return sampled_color.detach().cpu().numpy()

    def get_vertices_normal(self, vertices):
        normals = self.sdf_network.gradient(vertices).reshape(-1, 3)
        return normals.detach().cpu().numpy()

    def get_vertices_sdf(self, vertices):
        normals = self.sdf_network.sdf(vertices)
        return normals.detach().cpu().numpy()

    def validate_normal(self, num_steps=10, world_space=True, patch_size=10000, num_points=900000, disturb=0.01):
        point_cloud, duration = self.renderer.generate_point_cloud(num_steps=num_steps, device=self.device, num_points=num_points)
        point_cloud = (np.random.rand(len(point_cloud), 3) * 2 - 1) * disturb + point_cloud

        print('num_steps', num_steps, 'duration', duration)

        normal_cpu = []
        while len(normal_cpu) < len(point_cloud):
            print(len(normal_cpu))
            min_idx = len(normal_cpu)
            max_idx = np.min([len(point_cloud), min_idx+patch_size])
            normal = self.get_vertices_normal(vertices=torch.from_numpy(point_cloud[min_idx:max_idx, :].astype(np.float32)).to(self.device))
            if len(normal_cpu)==0:
                normal_cpu = normal
            else:
                normal_cpu = np.vstack((normal_cpu, normal))

        if world_space:
            point_cloud = point_cloud * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        normal_cpu_p = normal_cpu.clip(0, 1)
        normal_cpu_n = -normal_cpu.clip(-1, 0)
        normal_cpu = (normal_cpu_p + normal_cpu_n[:, [1, 2, 0]])*255
        normal_cpu = np.clip(normal_cpu, 0, 255)
        cloud = trimesh.points.PointCloud(vertices=point_cloud, colors=normal_cpu)
        cloud.export(os.path.join(self.base_exp_dir, 'meshes', 'pn{:0>8d}.ply'.format(self.iter_step)))

    def validate_udf(self, num_steps=10, world_space=True, patch_size=10000, num_points=900000, disturb=0.01):
        point_cloud, duration = self.renderer.generate_point_cloud(num_steps=num_steps, device=self.device, num_points=num_points)
        point_cloud = (np.random.rand(len(point_cloud), 3) * 2 - 1) * disturb + point_cloud

        print('num_steps', num_steps, 'duration', duration)

        sdf_cpu = []
        while len(sdf_cpu) < len(point_cloud):
            print(len(sdf_cpu))
            min_idx = len(sdf_cpu)
            max_idx = np.min([len(point_cloud), min_idx+patch_size])
            sdf = self.get_vertices_sdf(vertices=torch.from_numpy(point_cloud[min_idx:max_idx, :].astype(np.float32)).to(self.device))
            if len(sdf_cpu)==0:
                sdf_cpu = sdf
            else:
                sdf_cpu = np.vstack((sdf_cpu, sdf))

        if world_space:
            point_cloud = point_cloud * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        sdf_cpu = np.column_stack((np.zeros_like(sdf_cpu), (sdf_cpu*100).clip(0, 1)*255, (sdf_cpu*100).clip(0, 1)*255))
        cloud = trimesh.points.PointCloud(vertices=point_cloud, colors=sdf_cpu)
        cloud.export(os.path.join(self.base_exp_dir, 'meshes', 'pu{:0>8d}.ply'.format(self.iter_step)))

    def get_udf_normals_grid_slow(self, b_min, b_max, N=56, max_batch=int(2 ** 20), fourier=False):
        """
        Fills a dense N*N*N regular grid by querying the decoder network
        Inputs:
            decoder: coordinate network to evaluate
            latent_vec: conditioning vector
            N: grid size
            max_batch: number of points we can simultaneously evaluate
            fourier: are xyz coordinates encoded with fourier?
        Returns:
            df_values: (N,N,N) tensor representing distance field values on the grid
            vecs: (N,N,N,3) tensor representing gradients values on the grid, only for locations with a small
                    distance field value
            samples: (N**3, 7) tensor representing (x,y,z, distance field, grad_x, grad_y, grad_z)
        """
        self.sdf_network.eval()
        ################
        # 1: setting up the empty grid
        ################
        # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
        voxel_origin = [b_min, b_min, b_min]
        voxel_size = (b_max - b_min) / (N - 1)
        overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
        samples = torch.zeros(N ** 3, 7).cpu()
        # transform first 3 columns
        # to be the x, y, z index
        samples[:, 2] = overall_index % N
        samples[:, 1] = torch.div(overall_index, N, rounding_mode='floor') % N
        samples[:, 0] = torch.div(torch.div(overall_index, N, rounding_mode='floor'), N, rounding_mode='floor') % N
        # transform first 3 columns
        # to be the x, y, z coordinate
        samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
        samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
        samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]
        num_samples = N ** 3
        samples.requires_grad = False
        # samples.pin_memory()
        ################
        # 2: Run forward pass to fill the grid
        ################
        head = 0
        ## FIRST: fill distance field grid without gradients
        while head < num_samples:
            # xyz coords
            sample_subset = samples[head: min(head + max_batch, num_samples), 0:3].clone().cuda()
            # Create input
            if fourier:
                xyz = ws.fourier_transform(sample_subset)
            else:
                xyz = sample_subset
            # Run forward pass
            with torch.no_grad():
                df = self.sdf_network.sdf(xyz)
            # Store df
            samples[head: min(head + max_batch, num_samples), 3] = df.squeeze(-1).detach().cpu()
            # Next iter
            head += max_batch
        #
        ## THEN: compute gradients only where needed,
        # ie. where the predicted df value is small
        max_batch = max_batch // 4
        norm_mask = samples[:, 3] < 2 * voxel_size
        norm_idx = torch.where(norm_mask)[0]
        head, num_samples = 0, norm_idx.shape[0]
        while head < num_samples:
            # Find the subset of indices to compute:
            # -> a subset of indices where normal computations are needed
            sample_subset_mask = torch.zeros_like(norm_mask)
            sample_subset_mask[norm_idx[head]: norm_idx[min(head + max_batch, num_samples) - 1] + 1] = True
            sample_subset_mask = norm_mask * sample_subset_mask
            # xyz coords
            sample_subset = samples[sample_subset_mask, 0:3].clone().cuda()
            sample_subset.requires_grad = True
            # Create input
            if fourier:
                xyz = ws.fourier_transform(sample_subset)
            else:
                xyz = sample_subset
            # Run forward pass
            df = self.sdf_network.sdf(xyz)
            # Compute and store normalized vectors pointing towards the surface
            df.sum().backward()
            grad = sample_subset.grad.detach()
            samples[sample_subset_mask, 4:] = - F.normalize(grad, dim=1).cpu()
            # Next iter
            head += max_batch
        #
        # Separate values in DF / gradients
        df_values = samples[:, 3]
        df_values = df_values.reshape(N, N, N)
        vecs = samples[:, 4:]
        vecs = vecs.reshape(N, N, N, 3)
        return df_values, vecs, samples

    def get_mesh_udf_fast(self, b_min, b_max, N_MC=128, fourier=False,
                          gradient=True, eps=0.1, border_gradients=False, smooth_borders=False):
        """
        Computes a triangulated mesh from a distance field network conditioned on the latent vector
        Inputs:
            decoder: coordinate network to evaluate
            latent_vec: conditioning vector
            samples: already computed (N**3, 7) tensor representing (x,y,z, distance field, grad_x, grad_y, grad_z)
                        for a previous latent_vec, which is assumed to be close to the current one, if any
            indices: tensor representing the coordinates that need updating in the previous samples tensor (to speed
                        up iterations)
            N_MC: grid size
            fourier: are xyz coordinates encoded with fourier?
            gradient: do we need gradients?
            eps: length of the normal vectors used to derive gradients
            border_gradients: add a special case for border gradients?
            smooth_borders: do we smooth borders with a Laplacian?
        Returns:
            verts: vertices of the mesh
            faces: faces of the mesh
            mesh: trimesh object of the mesh
            samples: (N**3, 7) tensor representing (x,y,z, distance field, grad_x, grad_y, grad_z)
            indices: tensor representing the coordinates that need updating in the next iteration
        """
        ### 1: sample grid
        df_values, normals, samples = self.get_udf_normals_grid_slow(b_min, b_max, N=N_MC, fourier=fourier)
        df_values[df_values < 0] = 0
        ### 2: run our custom MC on it
        N = df_values.shape[0]
        voxel_size = (b_max - b_min) / (N - 1)
        voxel_origin = [b_min, b_min, b_min]
        verts, faces, _, _ = udf_mc_lewiner(df_values.cpu().detach().numpy(),
                                            normals.cpu().detach().numpy(),
                                            spacing=[voxel_size] * 3)
        verts = verts + b_min  # since voxel_origin = [-1, -1, -1]
        ### 3: evaluate vertices DF, and remove the ones that are too far
        verts_torch = torch.from_numpy(verts).float().cuda()
        with torch.no_grad():
            if fourier:
                xyz = ws.fourier_transform(verts_torch)
            else:
                xyz = verts_torch
            pred_df_verts = self.sdf_network.sdf(xyz)
        pred_df_verts = pred_df_verts.cpu().numpy()
        # Remove faces that have vertices far from the surface
        filtered_faces = faces[np.max(pred_df_verts[faces], axis=1)[:, 0] < voxel_size / 6]
        filtered_mesh = trimesh.Trimesh(verts, filtered_faces)
        ### 4: clean the mesh a bit
        # Remove NaNs, flat triangles, duplicate faces
        filtered_mesh = filtered_mesh.process(
            validate=False)  # DO NOT try to consistently align winding directions: too slow and poor results
        filtered_mesh.remove_duplicate_faces()
        filtered_mesh.remove_degenerate_faces()
        # Fill single triangle holes
        filtered_mesh.fill_holes()

        filtered_mesh_2 = trimesh.Trimesh(filtered_mesh.vertices, filtered_mesh.faces)
        # Re-process the mesh until it is stable:
        n_verts, n_faces, n_iter = 0, 0, 0
        while (n_verts, n_faces) != (len(filtered_mesh_2.vertices), len(filtered_mesh_2.faces)) and n_iter < 10:
            filtered_mesh_2 = filtered_mesh_2.process(validate=False)
            filtered_mesh_2.remove_duplicate_faces()
            filtered_mesh_2.remove_degenerate_faces()
            (n_verts, n_faces) = (len(filtered_mesh_2.vertices), len(filtered_mesh_2.faces))
            n_iter += 1
            filtered_mesh_2 = trimesh.Trimesh(filtered_mesh_2.vertices, filtered_mesh_2.faces)

        filtered_mesh = trimesh.Trimesh(filtered_mesh_2.vertices, filtered_mesh_2.faces)

        if smooth_borders:
            # Identify borders: those appearing only once
            border_edges = trimesh.grouping.group_rows(filtered_mesh.edges_sorted, require_count=1)

            # Build a dictionnary of (u,l): l is the list of vertices that are adjacent to u
            neighbours = defaultdict(lambda: [])
            for (u, v) in filtered_mesh.edges_sorted[border_edges]:
                neighbours[u].append(v)
                neighbours[v].append(u)
            border_vertices = np.array(list(neighbours.keys()))

            # Build a sparse matrix for computing laplacian
            pos_i, pos_j = [], []
            for k, ns in enumerate(neighbours.values()):
                for j in ns:
                    pos_i.append(k)
                    pos_j.append(j)

            sparse = coo_matrix((np.ones(len(pos_i)),  # put ones
                                 (pos_i, pos_j)),  # at these locations
                                shape=(len(border_vertices), len(filtered_mesh.vertices)))

            # Smoothing operation:
            lambda_ = 0.3
            for _ in range(5):
                border_neighbouring_averages = sparse @ filtered_mesh.vertices / sparse.sum(axis=1)
                laplacian = border_neighbouring_averages - filtered_mesh.vertices[border_vertices]
                filtered_mesh.vertices[border_vertices] = filtered_mesh.vertices[border_vertices] + lambda_ * laplacian

        if not gradient:
            return torch.tensor(filtered_mesh.vertices).float().cuda(), torch.tensor(
                filtered_mesh.faces).long().cuda(), filtered_mesh
        else:
            ### 5: use the mesh to compute normals
            normals = trimesh.geometry.weighted_vertex_normals(vertex_count=len(filtered_mesh.vertices),
                                                               faces=filtered_mesh.faces,
                                                               face_normals=filtered_mesh.face_normals,
                                                               face_angles=filtered_mesh.face_angles)
            ### 6: evaluate the DF around each vertex, based on normals
            normals = torch.tensor(normals).float().cuda()
            verts = torch.tensor(filtered_mesh.vertices).float().cuda()
            if fourier:
                xyz_s1 = ws.fourier_transform(verts + eps * normals)
                xyz_s2 = ws.fourier_transform(verts - eps * normals)
            else:
                xyz_s1 = verts + eps * normals
                xyz_s2 = verts - eps * normals
            s1 = decoder(torch.cat([latent_vec.repeat(verts.shape[0], 1), xyz_s1], dim=1))
            s2 = decoder(torch.cat([latent_vec.repeat(verts.shape[0], 1), xyz_s2], dim=1))
            # Re-plug differentiability here, by this rewriting trick
            new_verts = verts - eps * s1 * normals + eps * s2 * normals

            ## Compute indices needed for re-evaluation at the next iteration
            # fetch bins that are activated
            k = ((new_verts[:, 2].detach().cpu().numpy() - voxel_origin[2]) / voxel_size).astype(int)
            j = ((new_verts[:, 1].detach().cpu().numpy() - voxel_origin[1]) / voxel_size).astype(int)
            i = ((new_verts[:, 0].detach().cpu().numpy() - voxel_origin[0]) / voxel_size).astype(int)
            # find points around
            next_samples = i * N_MC * N_MC + j * N_MC + k
            next_samples_ip = np.minimum(i + 1, N_MC - 1) * N_MC * N_MC + j * N_MC + k
            next_samples_jp = i * N_MC * N_MC + np.minimum(j + 1, N_MC - 1) * N_MC + k
            next_samples_kp = i * N_MC * N_MC + j * N_MC + np.minimum(k + 1, N - 1)
            next_samples_im = np.maximum(i - 1, 0) * N_MC * N_MC + j * N_MC + k
            next_samples_jm = i * N_MC * N_MC + np.maximum(j - 1, 0) * N_MC + k
            next_samples_km = i * N_MC * N_MC + j * N_MC + np.maximum(k - 1, 0)
            # Concatenate
            next_indices = np.concatenate((next_samples, next_samples_ip, next_samples_jp,
                                           next_samples_kp, next_samples_im, next_samples_jm, next_samples_km))

            if border_gradients:
                ### 7: Add gradients at the surface borders?
                # Identify borders
                border_edges = trimesh.grouping.group_rows(filtered_mesh.edges_sorted, require_count=1)

                # Build a dictionnary of (u,v) edges, such that each vertex on the border
                # gets associated to exactly one border edge
                border_edges_dict = {}
                for (u, v) in filtered_mesh.edges_sorted[border_edges]:
                    border_edges_dict[u] = v
                    border_edges_dict[v] = u
                u_v_border = np.array(list(border_edges_dict.items()))
                u_border = u_v_border[:, 0]  # split border edges (u,v) into u and v arrays
                v_border = u_v_border[:, 1]

                # For each vertex on the border, take the cross product between
                # its normal and the border's edge
                normals_border = normals[u_border]
                edge_border = filtered_mesh.vertices[v_border] - filtered_mesh.vertices[u_border]
                edge_border = torch.tensor(edge_border).float().cuda()
                out_vec = torch.cross(edge_border, normals_border, dim=1)
                out_vec = out_vec / (torch.norm(out_vec, dim=1, keepdim=True) + 1e-6)  # make it unit length

                # Then we need to orient the out_vec such that they point outwards
                # To do so, we evaluate at +- their offset, and take the corresponding max DF value
                border_verts = torch.tensor(filtered_mesh.vertices[u_border]).float().cuda()
                if fourier:
                    xyz_s1_border = ws.fourier_transform(border_verts + 3 * eps * out_vec)
                    xyz_s2_border = ws.fourier_transform(border_verts - 3 * eps * out_vec)
                else:
                    xyz_s1_border = border_verts + 3 * eps * out_vec
                    xyz_s2_border = border_verts - 3 * eps * out_vec

                s1_border = decoder(torch.cat([latent_vec.repeat(border_verts.shape[0], 1), xyz_s1_border], dim=1))
                s2_border = decoder(torch.cat([latent_vec.repeat(border_verts.shape[0], 1), xyz_s2_border], dim=1))
                s1s2 = torch.stack((s1_border, s2_border))
                sign_out_vec = -torch.argmax(s1s2, dim=0) * 2 + 1
                out_vec = sign_out_vec * out_vec

                # Filter out the verts borders for which a displacement of out_vec
                # still evaluates at < eps DF, ie. verts classified as borders which are not really so
                u_border_filtered = u_border[((s1_border + s2_border)[:, 0] > eps).detach().cpu().numpy()]
                out_vec_filtered = out_vec[(s1_border + s2_border)[:, 0] > eps]
                out_df_filtered = torch.max(s1_border, s2_border)[(s1_border + s2_border) > eps]

                # Plug gradients to verts positions
                s_border = (eps * (out_df_filtered - out_df_filtered.detach())).unsqueeze(
                    -1)  # Fake zero, just to pass grads
                new_verts[u_border_filtered] = new_verts[u_border_filtered] - s_border * out_vec_filtered

            return new_verts, torch.tensor(filtered_mesh.faces).long().cuda(), filtered_mesh, samples, next_indices

    def validate_mesh_udf(self, world_space=False, resolution=64, color=False):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)
        vertices, triangles, vertices_color = self.validate_mesh_udf_main(bound_min, bound_max, resolution, color)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        if color:
            mesh.visual.vertex_colors = vertices_color
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', 'mu{:0>8d}.ply'.format(self.iter_step)))

        logging.info('End')

    def validate_mesh_udf_main(self, b_min, b_max, n, color):
        b_min_np = b_min.detach().cpu().numpy().min()
        b_max_np = b_max.detach().cpu().numpy().max()
        vertices, triangles, _ = self.get_mesh_udf_fast(b_min=b_min_np, b_max=b_max_np,
                                                        N_MC=n, gradient=False,
                                                        smooth_borders=False, fourier=False)
        if color:
            vertices_color = self.get_vertices_color(vertices)
        else:
            vertices_color = torch.zeros_like(vertices)
        return vertices.detach().cpu().numpy(), triangles.detach().cpu().numpy(), vertices_color.detach().cpu().numpy()[
                                                                                  :, ::-1]



if __name__ == '__main__':
    print('Hello Wooden')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0025)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--idx', type=int, default=-1)
    parser.add_argument('--is_sequence', default=False, action='store_true')
    parser.add_argument('--unit_sphere', default=False, action='store_true')
    parser.add_argument('--rev_sequence', default=False, action='store_true')
    parser.add_argument('--use_color', default=False, action='store_true')

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    if not args.is_sequence:
        runner = Runner(args.conf, args.mode, args.case, args.is_continue)

        if args.mode == 'train':
            runner.train()
        elif args.mode == 'validate_mesh':
            runner.validate_mesh(world_space=not args.unit_sphere, resolution=args.resolution, threshold=args.mcube_threshold)
        elif args.mode == 'validate_point':
            runner.validate_point(10, color=args.use_color, world_space=not args.unit_sphere)
        elif args.mode.startswith('interpolate'):  # Interpolate views given two image indices
            _, img_idx_0, img_idx_1 = args.mode.split('_')
            img_idx_0 = int(img_idx_0)
            img_idx_1 = int(img_idx_1)
            runner.interpolate_view(img_idx_0, img_idx_1)
        elif args.mode == 'validate_mesh_udf':
            runner.validate_mesh_udf(world_space=not args.unit_sphere, resolution=args.resolution, color=args.use_color)
        elif args.mode == 'refine':
            runner.refine()
        elif args.mode == 'validate_image':
            runner.validate_image(idx=args.idx)
        elif args.mode == 'validate_ray':
            runner.renderer.iter_step = runner.iter_step
            runner.validate_ray()
        elif args.mode == 'validate_normal':
            runner.validate_normal()
        elif args.mode == 'validate_udf':
            runner.validate_udf()
        elif args.mode == 'validate_mesh_poisson':
            runner.validate_mesh_poisson()
        elif args.mode == 'validate_mesh_poisson_test':
            runner.validate_mesh_poisson_test(threshold=args.mcube_threshold)
    else:
        case_list = os.listdir(os.path.join('./public_data', args.case))
        if args.rev_sequence:
            case_list = reversed(case_list)
        for case in case_list:
            runner = Runner(args.conf, args.mode, os.path.join(args.case, case), args.is_continue)

            if args.mode == 'train':
                runner.train()
                runner.validate_point()
                # try:
                #     runner.validate_mesh_udf(world_space=True, resolution=args.resolution)
                # except:
                #     print('validate_mesh_udf error')
                # finally:
                #     pass
            elif args.mode == 'validate_mesh':
                runner.validate_mesh(world_space=True, resolution=args.resolution, threshold=args.mcube_threshold)
            elif args.mode == 'validate_point':
                runner.validate_point(10, color=args.use_color, world_space=not args.unit_sphere)
            elif args.mode.startswith('interpolate'):  # Interpolate views given two image indices
                _, img_idx_0, img_idx_1 = args.mode.split('_')
                img_idx_0 = int(img_idx_0)
                img_idx_1 = int(img_idx_1)
                runner.interpolate_view(img_idx_0, img_idx_1)
            elif args.mode == 'validate_mesh_udf':
                runner.validate_mesh_udf(world_space=True, resolution=args.resolution, color=args.use_color)
            elif args.mode == 'refine':
                runner.refine()
            elif args.mode == 'validate_image':
                runner.validate_image(idx=args.idx)
            elif args.mode == 'validate_ray':
                runner.renderer.iter_step = runner.iter_step
                runner.validate_ray()
            elif args.mode == 'validate_normal':
                runner.validate_normal()
            elif args.mode == 'validate_udf':
                runner.validate_udf()

            del runner.dataset
            del runner

