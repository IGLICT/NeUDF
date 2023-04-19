import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import mcubes
from icecream import ic
import time
from skimage.measure import marching_cubes


def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    #print('pts shape: {}'.format(pts.shape))
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    print('threshold: {}'.format(threshold))

    #print('bound_min: {}'.format(bound_min))
    #print('bound_max: {}'.format(bound_max))

    u = extract_fields(bound_min, bound_max, resolution, query_func)

    # for debug udf
    u_min = np.min(u)
    u_max = np.max(u)
    print('u max: {}'.format(u_max))
    print('u min: {}'.format(u_min))

    if threshold < np.min(u) or threshold > np.max(u):
        threshold_mc = 0.99*u_min+0.01*u_max
    else:
        threshold_mc = threshold
    vertices, triangles, _, _ = marching_cubes(u, threshold_mc)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class NeuSRenderer:
    def __init__(self,
                 nerf,
                 sdf_network,
                 deviation_network,
                 color_network,
                 n_samples,
                 n_importance,
                 n_outside,
                 up_sample_steps,
                 perturb,
                 disturb=0.,
                 disturb_end=100000,
                 up_sample_mode='naive_appr',
                 up_sample_appr_level=0,
                 up_sample_s=64,
                 up_sample_min=1e-2,
                 render_mode='div',
                 ):
        self.nerf = nerf
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb
        self.disturb = disturb
        self.disturb_end = disturb_end
        self.up_sample_mode = up_sample_mode
        self.up_sample_appr_level = up_sample_appr_level
        self.up_sample_s = up_sample_s
        self.up_sample_min = up_sample_min
        self.render_mode = render_mode

    def render_core_outside(self, rays_o, rays_d, z_vals, sample_dist, nerf, background_rgb=None):
        """
        Render background
        """
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # batch_size, n_samples, 3

        dis_to_center = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).clip(1.0, 1e10)
        pts = torch.cat([pts / dis_to_center, 1.0 / dis_to_center], dim=-1)       # batch_size, n_samples, 4

        dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)

        pts = pts.reshape(-1, 3 + int(self.n_outside > 0))
        dirs = dirs.reshape(-1, 3)

        density, sampled_color = nerf(pts, dirs)
        alpha = 1.0 - torch.exp(-F.softplus(density.reshape(batch_size, n_samples)) * dists)
        alpha = alpha.reshape(batch_size, n_samples)
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
        color = (weights[:, :, None] * sampled_color).sum(dim=1)
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

        return {
            'color': color,
            'sampled_color': sampled_color,
            'alpha': alpha,
            'weights': weights,
        }

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, i):
        """
        Up sampling give a fixed inv_s
        """
        inv_s = 64 * 2 ** i
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        # prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        # next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        # prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        # next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        
        prev_esti_sdf = (mid_sdf - cos_val * dist * 0.5).clip(0., 1e6)
        next_esti_sdf = (mid_sdf + cos_val * dist * 0.5).clip(0., 1e6)
        prev_cdf = prev_esti_sdf*inv_s/(1+prev_esti_sdf*inv_s)
        next_cdf = next_esti_sdf*inv_s/(1+next_esti_sdf*inv_s)
        alpha = ((torch.abs(prev_cdf - next_cdf) +0) / (torch.abs(prev_cdf) + 1e-20)).clip(0., 1.)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        # print(weights.min(), alpha.min(), alpha.max())
        # if weights.min() < 0:
        #     exit()

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        # print(z_vals)
        # print(new_z_vals)
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    def render_core(self,
                    near,
                    far,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    sdf_network,
                    deviation_network,
                    color_network,
                    background_alpha=None,
                    background_sampled_color=None,
                    background_rgb=None,
                    cos_anneal_ratio=0.0,
                    weight_anneal_ratio=0.0,):
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        sdf_nn_output = sdf_network(pts)
        sdf = sdf_nn_output[:, :1]
        feature_vector = sdf_nn_output[:, 1:]

        #print('sdf max: {}'.format(torch.max(sdf)))
        #print('sdf min: {}'.format(torch.min(sdf)))

        if color_network.mode == 'second_order_udf':
            second_order_gradients, gradients = sdf_network.second_order_gradient(pts)
            second_order_gradients = second_order_gradients.squeeze()
            gradients = gradients.squeeze()
        else:
            second_order_gradients = None
            gradients = sdf_network.gradient(pts).squeeze()

        gradients_original = gradients

        if color_network.mode == 'normal_appr':
            sdf_bn = sdf.reshape(batch_size, n_samples)
            grad_bn = gradients.reshape(batch_size, n_samples, 3)
            grad_norm_bn = torch.linalg.norm(grad_bn, ord=2, dim=-1)
            is_sdf_small = (sdf_bn < 0.01)
            is_grad_small = (grad_norm_bn < 0.1)

            prev_grad_bn = grad_bn
            next_grad_bn = grad_bn
            prev_grad_sum = torch.zeros_like(grad_bn)
            next_grad_sum = torch.zeros_like(grad_bn)
            for _ in range(5):
                prev_grad_bn = torch.cat([prev_grad_bn[:, :1, :], prev_grad_bn[:, :-1, :]], dim=1)
                next_grad_bn = torch.cat([next_grad_bn[:, 1:, :], next_grad_bn[:, -1:, :]], dim=1)
                prev_grad_sum = prev_grad_sum + prev_grad_bn
                next_grad_sum = next_grad_sum + next_grad_bn
            grad_sum_delta = (prev_grad_sum * next_grad_sum).sum(-1)
            is_grad_singular = (grad_sum_delta < 0)

            is_normal_appr = (is_sdf_small & is_grad_small & is_grad_singular)
            # mid_z_vals_bn = mid_z_vals.reshape(batch_size, n_samples)
            # mid_z_vals_bn_delta = mid_z_vals_bn - 0.01
            # pts_bn_delta = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals_bn_delta[..., :, None]
            pts_bn = pts.reshape(batch_size, n_samples, 3)
            dirs_bn = F.normalize(prev_grad_bn, dim=-1).reshape(batch_size, n_samples, 3)
            pts_bn_delta = pts_bn + dirs_bn * 0.01
            pts_appr = pts_bn_delta[is_normal_appr, :]
            grad_appr = sdf_network.gradient(pts_appr.reshape(-1, 3)).squeeze()
            grad_bn[is_normal_appr, :] = grad_appr
            gradients = grad_bn.reshape(-1, 3)
            # print(grad_appr.shape)






        sampled_color = color_network(pts, gradients, dirs, feature_vector, sdf, second_order_gradients).reshape(batch_size, n_samples, 3)

        # inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
        inv_s = deviation_network(torch.zeros([1, 3]))[:, :1]
        inv_s = inv_s.expand(batch_size * n_samples, 1)
        # inv_s = inv_s.squeeze()
        # print(inv_s,inv_s.shape)

        true_cos = (dirs * gradients).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        true_cos_sign = torch.where(true_cos > 0, 1, -1)
        iter_cos = ((true_cos * 0.5 + 0.5 * true_cos_sign) * (1.0 - cos_anneal_ratio) +
                     (true_cos) * cos_anneal_ratio)  # not always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = (sdf + iter_cos * dists.reshape(-1, 1) * 0.5).clip(0., 1e6)
        estimated_prev_sdf = (sdf - iter_cos * dists.reshape(-1, 1) * 0.5).clip(0., 1e6)

        #print('estimated_next_sdf max: {}'.format(torch.max(estimated_next_sdf)))
        #print('estimated_next_sdf min: {}'.format(torch.min(estimated_next_sdf)))
        #print('estimated_prev_sdf max: {}'.format(torch.max(estimated_prev_sdf)))
        #print('estimated_prev_sdf min: {}'.format(torch.min(estimated_prev_sdf)))

        if self.render_mode == 'div' or 'div_anneal':
            prev_cdf = (estimated_prev_sdf * inv_s / (1+estimated_prev_sdf * inv_s)).clip(0., 1e6)
            next_cdf = (estimated_next_sdf * inv_s / (1+estimated_next_sdf * inv_s)).clip(0., 1e6)
        elif self.render_mode == 'exp':
            prev_cdf = (1 - torch.exp(-estimated_prev_sdf * inv_s)).clip(0., 1e6)
            next_cdf = (1 - torch.exp(-estimated_next_sdf * inv_s)).clip(0., 1e6)
        elif self.render_mode == 'atan':
            prev_cdf = (torch.atan(estimated_prev_sdf * inv_s) * 2 / np.pi).clip(0., 1e6)
            next_cdf = (torch.atan(estimated_next_sdf * inv_s) * 2 / np.pi).clip(0., 1e6)
        elif self.render_mode == 'neus':
            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
        elif self.render_mode == 'neus_abs':
            prev_cdf = torch.abs(torch.sigmoid(estimated_prev_sdf * inv_s))
            next_cdf = torch.abs(torch.sigmoid(estimated_next_sdf * inv_s))
        elif self.render_mode == 'div_appr':
            prev_cdf = ((0.1+estimated_prev_sdf * inv_s) / (1+estimated_prev_sdf * inv_s)).clip(0., 1e6)
            next_cdf = ((0.1+estimated_next_sdf * inv_s) / (1+estimated_next_sdf * inv_s)).clip(0., 1e6)
        
        p = torch.abs(prev_cdf - next_cdf)
        c = torch.max(prev_cdf, next_cdf)

        alpha = ((p + 0) / (c + 1e-10)).reshape(batch_size, n_samples).clip(0.0, 1.0)

        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < 1.0).float().detach()
        relax_inside_sphere = (pts_norm < 1.2).float().detach()

        # Render with background
        if background_alpha is not None:
            alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * (1.0 - inside_sphere)
            alpha = torch.cat([alpha, background_alpha[:, n_samples:]], dim=-1)
            sampled_color = sampled_color * inside_sphere[:, :, None] +\
                            background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
            sampled_color = torch.cat([sampled_color, background_sampled_color[:, n_samples:]], dim=1)

        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        if self.render_mode == 'div_anneal':
            weights = (weights * 0.5 + 0.5 / n_samples) * (1. - weight_anneal_ratio) + weights * weight_anneal_ratio

        weights_sum = weights.sum(dim=-1, keepdim=True)

        color = (sampled_color * weights[:, :, None]).sum(dim=1)

        if background_rgb is not None:    # Fixed background, usually black
            color = color + background_rgb * (1.0 - weights_sum)


        # Eikonal loss
        gradient_error = (torch.linalg.norm(gradients_original.reshape(batch_size, n_samples, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
        # gradient_error = gradient_error * dists * 64 / 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)


        # normal loss
        normals = gradients.reshape(batch_size, n_samples, 3)
        normals = normals / torch.linalg.norm(normals, ord=2, dim=-1, keepdim=True)
        prev_normals = torch.cat([normals[:, :1, :], normals[:, :-1, :]], dim=1)
        sin_p = torch.abs((normals-prev_normals)).sum(dim=-1)
        sin_n = torch.abs((normals+prev_normals)).sum(dim=-1)
        sin = torch.min(sin_p, sin_n)
        normal_error = torch.abs(sin)
        normal_error = (normal_error * relax_inside_sphere).sum() / (relax_inside_sphere.sum() + 1e-5)

        return {
            'color': color,
            'dists': dists,
            'gradients': gradients.reshape(batch_size, n_samples, 3),
            's_val': 1.0 / inv_s,
            'mid_z_vals': mid_z_vals,
            'weights': weights,
            'cdf': c.reshape(batch_size, n_samples),
            'gradient_error': gradient_error,
            'inside_sphere': inside_sphere,
            'normal_error': normal_error,
        }

    def render(self, rays_o, rays_d, near, far, perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=0.0, weight_anneal_ratio=0.0, up_sample=False):
        z_dist_sdf_gradient = 0

        batch_size = len(rays_o)
        sample_dist = 2.0 / self.n_samples   # Assuming the region of interest is a unit sphere
        z_vals = torch.linspace(0.0, 1.0, self.n_samples)
        z_vals = near + (far - near) * z_vals[None, :]

        z_vals_outside = None
        if self.n_outside > 0:
            z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside)

        n_samples = self.n_samples
        perturb = self.perturb

        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1]) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

            if self.n_outside > 0:
                mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
                lower = torch.cat([z_vals_outside[..., :1], mids], -1)
                t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]])
                z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand

        if self.n_outside > 0:
            z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / self.n_samples

        background_alpha = None
        background_sampled_color = None

        # Up sample
        if up_sample and self.n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, self.n_samples)

                if self.up_sample_mode == 'surface':
                    max_r = 0.1
                    min_r = 0.005
                    beta = 1.5e-5
                    interval_radius = np.max([max_r * np.exp(-self.iter_step * beta), min_r])
                    new_z_vals, new_z_vals_inv = self.up_sample_surface(rays_o, rays_d, z_vals, sdf, self.n_importance//2, interval_radius)
                    z_vals, sdf = self.cat_z_vals(rays_o, rays_d, z_vals, new_z_vals, sdf, last=True)
                    z_vals, sdf = self.cat_z_vals(rays_o, rays_d, z_vals, new_z_vals_inv, sdf, last=True)
                elif self.up_sample_mode == 'naive':
                    for i in range(self.up_sample_steps):
                        new_z_vals = self.up_sample_naive(rays_o,
                                                    rays_d,
                                                    z_vals,
                                                    sdf,
                                                    self.n_importance // self.up_sample_steps,
                                                    i)

                        z_vals, sdf = self.cat_z_vals(rays_o,
                                                      rays_d,
                                                      z_vals,
                                                      new_z_vals,
                                                      sdf,
                                                      last=False)
                elif self.up_sample_mode == 'neus':
                    for i in range(self.up_sample_steps):
                        new_z_vals = self.up_sample(rays_o,
                                                          rays_d,
                                                          z_vals,
                                                          sdf,
                                                          self.n_importance // self.up_sample_steps,
                                                          i)

                        z_vals, sdf = self.cat_z_vals(rays_o,
                                                      rays_d,
                                                      z_vals,
                                                      new_z_vals,
                                                      sdf,
                                                      last=False)
                elif self.up_sample_mode == 'udf':
                    for i in range(self.up_sample_steps):
                        new_z_vals = self.up_sample_udf(rays_o,
                                                          rays_d,
                                                          z_vals,
                                                          sdf,
                                                          self.n_importance // self.up_sample_steps,
                                                          i)

                        z_vals, sdf = self.cat_z_vals(rays_o,
                                                      rays_d,
                                                      z_vals,
                                                      new_z_vals,
                                                      sdf,
                                                      last=False)
                elif self.up_sample_mode == 'neus_appr':
                    for i in range(self.up_sample_steps):
                        new_z_vals = self.up_sample_neus_appr(rays_o,
                                                          rays_d,
                                                          z_vals,
                                                          sdf,
                                                          self.n_importance // self.up_sample_steps,
                                                          i)

                        z_vals, sdf = self.cat_z_vals(rays_o,
                                                      rays_d,
                                                      z_vals,
                                                      new_z_vals,
                                                      sdf,
                                                      last=False)
                elif self.up_sample_mode == 'naive_appr':
                    for i in range(self.up_sample_steps):
                        new_z_vals = self.up_sample_naive_appr(rays_o,
                                                          rays_d,
                                                          z_vals,
                                                          sdf,
                                                          self.n_importance // self.up_sample_steps,
                                                          i)

                        z_vals, sdf = self.cat_z_vals(rays_o,
                                                      rays_d,
                                                      z_vals,
                                                      new_z_vals,
                                                      sdf,
                                                      last=False)
                elif self.up_sample_mode == 'fake_f':
                    for i in range(self.up_sample_steps):
                        new_z_vals = self.up_sample_fake_f(rays_o,
                                                          rays_d,
                                                          z_vals,
                                                          sdf,
                                                          self.n_importance // self.up_sample_steps,
                                                          i)

                        z_vals, sdf = self.cat_z_vals(rays_o,
                                                      rays_d,
                                                      z_vals,
                                                      new_z_vals,
                                                      sdf,
                                                      last=False)
                elif self.up_sample_mode == 'uniform':
                    for i in range(self.up_sample_steps):
                        new_z_vals = self.up_sample_uniform(rays_o,
                                                          rays_d,
                                                          z_vals,
                                                          sdf,
                                                          self.n_importance // self.up_sample_steps,
                                                          i)

                        z_vals, sdf = self.cat_z_vals(rays_o,
                                                      rays_d,
                                                      z_vals,
                                                      new_z_vals,
                                                      sdf,
                                                      last=False)

            n_samples = self.n_samples + self.n_importance

        # Background model
        if self.n_outside > 0:
            z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
            z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)
            ret_outside = self.render_core_outside(rays_o, rays_d, z_vals_feed, sample_dist, self.nerf)

            background_sampled_color = ret_outside['sampled_color']
            background_alpha = ret_outside['alpha']

        # Render core
        ret_fine = self.render_core(near,
                                    far,
                                    rays_o,
                                    rays_d,
                                    z_vals,
                                    sample_dist,
                                    self.sdf_network,
                                    self.deviation_network,
                                    self.color_network,
                                    background_rgb=background_rgb,
                                    background_alpha=background_alpha,
                                    background_sampled_color=background_sampled_color,
                                    cos_anneal_ratio=cos_anneal_ratio,
                                    weight_anneal_ratio=weight_anneal_ratio,)

        color_fine = ret_fine['color']
        weights = ret_fine['weights']
        weights_sum = weights.sum(dim=-1, keepdim=True)
        gradients = ret_fine['gradients']
        s_val = ret_fine['s_val'].reshape(batch_size, n_samples).mean(dim=-1, keepdim=True)

        return {
            'color_fine': color_fine,
            's_val': s_val,
            'cdf_fine': ret_fine['cdf'],
            'weight_sum': weights_sum,
            'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
            'gradients': gradients,
            'weights': weights,
            'gradient_error': ret_fine['gradient_error'],
            'inside_sphere': ret_fine['inside_sphere'],
            'normal_error': ret_fine['normal_error'],
        }

    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: self.sdf_network.sdf(pts))

    def generate_point_cloud(self, device, num_steps = 10, num_points = 900000, filter_val = 0.009, threshold=0.1):

        start = time.time()


        for param in self.sdf_network.parameters():
            param.requires_grad = False

        sample_num = 200000
        samples_cpu = np.zeros((0, 3))
        samples = torch.rand(1, sample_num, 3).float().to(device) * 2 - 1
        samples.requires_grad = True

        #encoding = self.model.encoder(inputs)

        i = 0
        while len(samples_cpu) < num_points:
            print('iteration', i)

            for j in range(num_steps):
                print('refinement', j)
                df_pred = torch.clamp(self.sdf_network.sdf(samples.squeeze(0)), max=threshold).unsqueeze(0)

                df_pred.sum().backward()

                gradient = samples.grad.detach()
                samples = samples.detach()
                df_pred = df_pred.detach()
                #inputs = inputs.detach()
                samples = samples - F.normalize(gradient, dim=2) * df_pred.reshape(-1, 1)  # better use Tensor.copy method?
                samples = samples.detach()
                samples.requires_grad = True


            print('finished refinement')

            if not i == 0:
                samples_cpu = np.vstack((samples_cpu, samples[df_pred.squeeze(-1) < filter_val].detach().cpu().numpy()))

            samples = samples[df_pred.squeeze(-1) < 0.03].unsqueeze(0)
            indices = torch.randint(samples.shape[1], (1, sample_num))
            samples = samples[[[0, ] * sample_num], indices]
            samples += (threshold / 3) * torch.randn(samples.shape).to(device)  # 3 sigma rule
            samples = samples.detach()
            samples.requires_grad = True

            i += 1
            print(samples_cpu.shape)

        duration = time.time() - start

        return samples_cpu, duration
    
    def up_sample_uniformity(self, rays_o, rays_d, z_vals, sdf, n_importance):
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)

        new_z_vals = torch.linspace(0. + 0.5 / n_importance, 1. - 0.5 / n_importance, steps=n_importance)
        new_z_vals = new_z_vals.expand([batch_size, n_importance])
        
        _, bins_index = torch.min(torch.max(sdf,100*(1-1*inside_sphere)), dim=-1)
        
        prev_z_val = z_vals[torch.arange(0, batch_size), (bins_index-1).clip(0, n_samples)]
        next_z_val = z_vals[torch.arange(0, batch_size), (bins_index+1).clip(0, n_samples)]
        
        new_z_vals = torch.ones_like(new_z_vals)*prev_z_val[:,None]+new_z_vals*(next_z_val-prev_z_val)[:,None]

        return new_z_vals.detach()

    def up_sample_surface(self, rays_o, rays_d, z_vals, sdf, n_importance, interval_radius):
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val * inside_sphere

        dist = (next_z_vals - prev_z_vals)

        inv_s = self.deviation_network(torch.zeros([1, 3]))[:, :1].detach()
        prev_esti_sdf = (mid_sdf - cos_val * dist * 0.5).clip(0., 1e6)
        next_esti_sdf = (mid_sdf + cos_val * dist * 0.5).clip(0., 1e6)
        prev_cdf = prev_esti_sdf * inv_s / (1 + prev_esti_sdf * inv_s)
        next_cdf = next_esti_sdf * inv_s / (1 + next_esti_sdf * inv_s)
        alpha = ((torch.abs(prev_cdf - next_cdf) + 0) / (torch.abs(prev_cdf) + 1e-20)).clip(0., 1.)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        _, min_weight_idx = torch.max(weights, dim=-1)
        prev_z_val = z_vals[torch.arange(0, batch_size), (min_weight_idx - 1).clip(0, n_samples - 1)]
        next_z_val = z_vals[torch.arange(0, batch_size), (min_weight_idx + 1).clip(0, n_samples - 1)]

        new_z_vals = torch.linspace(0. + 0.5 / n_importance, 1. - 0.5 / n_importance, steps=n_importance)
        new_z_vals = new_z_vals.expand([batch_size, n_importance])
        new_z_vals = torch.ones_like(new_z_vals) * prev_z_val[:, None] + new_z_vals * (next_z_val - prev_z_val)[:, None]

        new_pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        new_sdf = self.sdf_network.sdf(new_pts.reshape(-1, 3)).reshape(batch_size, n_importance)
        _, min_sdf_idx = torch.min(new_sdf, dim=-1)
        min_sdf_z = new_z_vals[torch.arange(0, batch_size), min_sdf_idx]
        sample_distribution = torch.abs(torch.randn(batch_size, n_importance)) * interval_radius
        sample_z_vals = sample_distribution + min_sdf_z[:, None]
        sample_z_vals_inv = -sample_distribution + min_sdf_z[:, None]

        new_pts_norm = torch.linalg.norm(new_pts.reshape(-1, 3), ord=2, dim=-1, keepdim=True).reshape(batch_size, -1)
        new_pts_inside_sphere = (new_pts_norm < 1.2).float()
        min_sdf_inside_sphere = new_pts_inside_sphere[torch.arange(0, batch_size), min_sdf_idx]

        uniform_sample_distribution = torch.linspace(0.+1./n_importance, 1.-1./n_importance, n_importance*2)
        uniform_sample_z_vals = z_vals[:, :1] + uniform_sample_distribution[None, :] * (z_vals[:, -1:]-z_vals[:, :1])
        uniform_sample_z_vals_inv = uniform_sample_z_vals[:, :n_importance]
        uniform_sample_z_vals = uniform_sample_z_vals[:, n_importance:]

        sample_z_vals = min_sdf_inside_sphere[:, None] * sample_z_vals + (1-min_sdf_inside_sphere)[:, None] * uniform_sample_z_vals
        sample_z_vals_inv = min_sdf_inside_sphere[:, None] * sample_z_vals_inv + (1-min_sdf_inside_sphere)[:, None] * uniform_sample_z_vals_inv

        return sample_z_vals.detach(), sample_z_vals_inv.detach()

    def up_sample_udf(self, rays_o, rays_d, z_vals, sdf, n_importance, i):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        mid_sdf = sdf[:, 1:]+sdf[:, :-1]

        inv_s = 64 * 2 ** i

        sigma = 1 / (1 + torch.exp(-mid_sdf * inv_s))
        rho = sigma * (1 - sigma) * inv_s
        weights = rho

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def up_sample_naive(self, rays_o, rays_d, z_vals, sdf, n_importance, i):
        """
        Up sampling give a fixed inv_s
        """
        with torch.enable_grad():
            batch_size, n_samples = z_vals.shape
            pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
            radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
            inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
            sdf = sdf.reshape(batch_size, n_samples)
            prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
            prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
            mid_sdf = (prev_sdf + next_sdf) * 0.5
            cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)
            
            inv_s = 64 * 2 ** i
        
            sigma = 1/(1+torch.exp(-mid_sdf * inv_s))
            rho = sigma * (1 - sigma) * inv_s
            

            alpha = 1 - torch.exp(-rho * (next_z_vals - prev_z_vals))
            # print('alpha',alpha[0])
            weights = alpha * torch.cumprod(
                torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

            z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def up_sample_naive_appr(self, rays_o, rays_d, z_vals, sdf, n_importance, i):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        inv_s = self.up_sample_s * 2 ** i

        sigma = 1 / (1 + torch.exp(-mid_sdf * inv_s))
        rho = sigma * (1 - sigma) * inv_s

        alpha = 1 - torch.exp(-rho * (next_z_vals - prev_z_vals))
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        for _ in range(self.up_sample_appr_level):
            weights_prev = torch.cat([weights[:, :1], weights[:, :-1]], dim=-1)
            weights_next = torch.cat([weights[:, 1:], weights[:, -1:]], dim=-1)
            weights = torch.max(weights, weights_prev)
            weights = torch.max(weights, weights_next)

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def up_sample_neus_appr(self, rays_o, rays_d, z_vals, sdf, n_importance, i):
        inv_s = 64 * 2 ** i
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val * inside_sphere

        dist = (next_z_vals - prev_z_vals)

        prev_esti_sdf = (mid_sdf - cos_val * dist * 0.5).clip(0., 1e6)
        next_esti_sdf = (mid_sdf + cos_val * dist * 0.5).clip(0., 1e6)
        prev_cdf = prev_esti_sdf * inv_s / (1 + prev_esti_sdf * inv_s)
        next_cdf = next_esti_sdf * inv_s / (1 + next_esti_sdf * inv_s)
        alpha = ((torch.abs(prev_cdf - next_cdf) + 0) / (torch.abs(prev_cdf) + 1e-20)).clip(0., 1.)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        for _ in range(3):
            weights_prev = torch.cat([weights[:, :1], weights[:, :-1]], dim=-1)
            weights_next = torch.cat([weights[:, 1:], weights[:, -1:]], dim=-1)
            weights = torch.max(weights, weights_prev)
            weights = torch.max(weights, weights_next)

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def up_sample_fake_f(self, rays_o, rays_d, z_vals, sdf, n_importance, i):
        """
        Up sampling give a fixed inv_s
        """
        inv_s = self.up_sample_s * 2 ** i
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        is_cos_inverse = torch.where(cos_val < 0, 1, -1)
        is_sdf_big = torch.where(sdf > 1e-2, 1, -1)
        is_fake_sdf = torch.max(torch.cat([torch.ones_like(is_cos_inverse[:, :1]), is_cos_inverse], dim=-1), is_sdf_big)
        fake_sdf = sdf * is_fake_sdf
        fake_cos_val = cos_val * is_cos_inverse

        fake_mid_sdf = (fake_sdf[:, 1:] + fake_sdf[:, :-1])/2
        dist = next_z_vals - prev_z_vals
        fake_prev_esti_sdf = fake_mid_sdf - fake_cos_val * dist * 0.5
        fake_next_esti_sdf = fake_mid_sdf + fake_cos_val * dist * 0.5
        fake_prev_cdf = torch.sigmoid(fake_prev_esti_sdf * inv_s)
        fake_next_cdf = torch.sigmoid(fake_next_esti_sdf * inv_s)

        alpha = ((torch.abs(fake_prev_cdf - fake_next_cdf) + 1e-10) / (torch.abs(fake_prev_cdf) + 1e-10)).clip(0., 1.)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def up_sample_uniform(self, rays_o, rays_d, z_vals, sdf, n_importance, i):
        inv_s = 64 * 2 ** i
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        cos_val = cos_val * inside_sphere

        dist = (next_z_vals - prev_z_vals)

        prev_esti_sdf = (mid_sdf - cos_val * dist * 0.5).clip(0., 1e6)
        next_esti_sdf = (mid_sdf + cos_val * dist * 0.5).clip(0., 1e6)
        prev_cdf = prev_esti_sdf * inv_s / (1 + prev_esti_sdf * inv_s)
        next_cdf = next_esti_sdf * inv_s / (1 + next_esti_sdf * inv_s)
        alpha = ((torch.abs(prev_cdf - next_cdf) + 0) / (torch.abs(prev_cdf) + 1e-20)).clip(0., 1.)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        weights = weights / (next_z_vals - prev_z_vals + 1e-5)

        for _ in range(self.up_sample_appr_level):
            weights_prev = torch.cat([weights[:, :1], weights[:, :-1]], dim=-1)
            weights_next = torch.cat([weights[:, 1:], weights[:, -1:]], dim=-1)
            weights = torch.max(weights, weights_prev)
            weights = torch.max(weights, weights_next)

        weights = weights * inside_sphere

        use_rand_us = ((weights.sum(dim=-1)) < 0.2).float()

        weights = weights / (mid_sdf + 1e-5)

        _, bin_idx = torch.max(weights, dim=-1)

        prev_z_val = z_vals[torch.arange(0, batch_size), (bin_idx - self.up_sample_appr_level).clip(0, n_samples-2)]
        next_z_val = z_vals[torch.arange(0, batch_size), (bin_idx + self.up_sample_appr_level + 1).clip(0, n_samples-2)]
        cos_val_mean = torch.abs(cos_val).sum(dim=-1)/(inside_sphere.sum(dim=-1)+1e-5)
        delta_z_val = (next_z_val - prev_z_val) / (cos_val_mean + 1e-5)
        delta_z_val = (delta_z_val - (next_z_val - prev_z_val)) / 2
        prev_z_val = prev_z_val - delta_z_val
        next_z_val = next_z_val + delta_z_val

        new_z_vals = torch.linspace(0. + 0.5 / n_importance, 1. - 0.5 / n_importance, steps=n_importance)
        new_z_vals = new_z_vals.expand([batch_size, n_importance])
        new_z_vals = torch.ones_like(new_z_vals) * prev_z_val[:, None] + new_z_vals * (next_z_val - prev_z_val)[:, None]

        begin_z_vals = z_vals[:, 0]
        end_z_vals = z_vals[:, -1]
        rand_z_vals = torch.rand(batch_size, n_importance)
        rand_z_vals = torch.ones_like(rand_z_vals) * begin_z_vals[:, None] + rand_z_vals * (end_z_vals - begin_z_vals)[:, None]

        new_z_vals = rand_z_vals * use_rand_us[:, None] + new_z_vals * (1 - use_rand_us[:, None])

        return new_z_vals.detach()

    def generate_ray(self, rays_o, rays_d, near, far, perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=0.0):
        batch_size = len(rays_o)
        sample_dist = 2.0 / self.n_samples   # Assuming the region of interest is a unit sphere
        z_vals = torch.linspace(0.0, 1.0, self.n_samples)
        z_vals = near + (far - near) * z_vals[None, :]

        z_vals_outside = None
        if self.n_outside > 0:
            z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside)

        n_samples = self.n_samples
        perturb = self.perturb

        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1]) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

            if self.n_outside > 0:
                mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
                lower = torch.cat([z_vals_outside[..., :1], mids], -1)
                t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]])
                z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand

        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
        sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, self.n_samples)

        # Up sample
        if self.n_importance > 0:
            with torch.no_grad():
                if self.up_sample_mode == 'surface':
                    max_r = 0.1
                    min_r = 0.005
                    beta = 1.5e-5
                    interval_radius = np.max([max_r * np.exp(-self.iter_step * beta), min_r])
                    new_z_vals, new_z_vals_inv = self.up_sample_surface(rays_o, rays_d, z_vals, sdf,
                                                                        self.n_importance // 2, interval_radius)
                    z_vals, sdf = self.cat_z_vals(rays_o, rays_d, z_vals, new_z_vals, sdf, last=True)
                    z_vals, sdf = self.cat_z_vals(rays_o, rays_d, z_vals, new_z_vals_inv, sdf, last=True)
                elif self.up_sample_mode == 'naive':
                    for i in range(self.up_sample_steps):
                        new_z_vals = self.up_sample_naive(rays_o,
                                                          rays_d,
                                                          z_vals,
                                                          sdf,
                                                          self.n_importance // self.up_sample_steps,
                                                          i)

                        z_vals, sdf = self.cat_z_vals(rays_o,
                                                      rays_d,
                                                      z_vals,
                                                      new_z_vals,
                                                      sdf,
                                                      last=False)
                elif self.up_sample_mode == 'neus':
                    for i in range(self.up_sample_steps):
                        new_z_vals = self.up_sample(rays_o,
                                                    rays_d,
                                                    z_vals,
                                                    sdf,
                                                    self.n_importance // self.up_sample_steps,
                                                    i)

                        z_vals, sdf = self.cat_z_vals(rays_o,
                                                      rays_d,
                                                      z_vals,
                                                      new_z_vals,
                                                      sdf,
                                                      last=False)
                elif self.up_sample_mode == 'udf':
                    for i in range(self.up_sample_steps):
                        new_z_vals = self.up_sample_udf(rays_o,
                                                        rays_d,
                                                        z_vals,
                                                        sdf,
                                                        self.n_importance // self.up_sample_steps,
                                                        i)

                        z_vals, sdf = self.cat_z_vals(rays_o,
                                                      rays_d,
                                                      z_vals,
                                                      new_z_vals,
                                                      sdf,
                                                      last=False)
                elif self.up_sample_mode == 'neus_appr':
                    for i in range(self.up_sample_steps):
                        new_z_vals = self.up_sample_neus_appr(rays_o,
                                                          rays_d,
                                                          z_vals,
                                                          sdf,
                                                          self.n_importance // self.up_sample_steps,
                                                          i)

                        z_vals, sdf = self.cat_z_vals(rays_o,
                                                      rays_d,
                                                      z_vals,
                                                      new_z_vals,
                                                      sdf,
                                                      last=False)
                elif self.up_sample_mode == 'naive_appr':
                    for i in range(self.up_sample_steps):
                        new_z_vals = self.up_sample_naive_appr(rays_o,
                                                          rays_d,
                                                          z_vals,
                                                          sdf,
                                                          self.n_importance // self.up_sample_steps,
                                                          i)

                        z_vals, sdf = self.cat_z_vals(rays_o,
                                                      rays_d,
                                                      z_vals,
                                                      new_z_vals,
                                                      sdf,
                                                      last=False)

        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        # cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere
        cos_val = cos_val * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = (mid_sdf - cos_val * dist * 0.5).clip(0., 1e6)
        next_esti_sdf = (mid_sdf + cos_val * dist * 0.5).clip(0., 1e6)
        inv_s = 64 * 2 ** (self.up_sample_steps - 1)
        prev_cdf = (prev_esti_sdf * inv_s / (1 + prev_esti_sdf * inv_s)).clip(0., 1e6)
        next_cdf = (next_esti_sdf * inv_s / (1 + next_esti_sdf * inv_s)).clip(0., 1e6)
        alpha = ((torch.abs(prev_cdf - next_cdf) + 1e-5) / (prev_cdf + 1e-5)).clip(0., 1.)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        return z_vals.detach().cpu().numpy().flatten(), \
               sdf.detach().cpu().numpy().flatten(), \
               weights.detach().cpu().numpy().flatten()

    def sample_symmetry(self, rays_o, rays_d, z_vals, sdf, n_importance, interval_radius):
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val * inside_sphere

        dist = (next_z_vals - prev_z_vals)

        inv_s = inv_s = self.deviation_network(torch.zeros([1, 3]))[:, :1].detach()
        prev_esti_sdf = (mid_sdf - cos_val * dist * 0.5).clip(0., 1e6)
        next_esti_sdf = (mid_sdf + cos_val * dist * 0.5).clip(0., 1e6)
        prev_cdf = prev_esti_sdf * inv_s / (1 + prev_esti_sdf * inv_s)
        next_cdf = next_esti_sdf * inv_s / (1 + next_esti_sdf * inv_s)
        alpha = ((torch.abs(prev_cdf - next_cdf) + 0) / (torch.abs(prev_cdf) + 1e-20)).clip(0., 1.)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        _, min_weight_idx = torch.max(weights, dim=-1)
        prev_z_val = z_vals[torch.arange(0, batch_size), (min_weight_idx - 1).clip(0, n_samples - 1)]
        next_z_val = z_vals[torch.arange(0, batch_size), (min_weight_idx + 1).clip(0, n_samples - 1)]

        new_z_vals = torch.linspace(0. + 0.5 / n_importance, 1. - 0.5 / n_importance, steps=n_importance)
        new_z_vals = new_z_vals.expand([batch_size, n_importance])
        new_z_vals = torch.ones_like(new_z_vals) * prev_z_val[:, None] + new_z_vals * (next_z_val - prev_z_val)[:, None]

        new_pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        new_sdf = self.sdf_network.sdf(new_pts.reshape(-1, 3)).reshape(batch_size, n_importance)
        _, min_sdf_idx = torch.min(new_sdf, dim=-1)
        min_sdf_z = new_z_vals[torch.arange(0, batch_size), min_sdf_idx]
        # sample_distribution = torch.randn(batch_size, n_importance) * interval_radius
        sample_distribution = torch.linspace(0. + 0.5 / n_importance, 1. - 0.5 / n_importance, steps=n_importance)
        sample_distribution = (sample_distribution * interval_radius).expand([batch_size, n_importance])
        sample_z_vals = sample_distribution + min_sdf_z[:, None]
        sample_z_vals_inv = -sample_distribution + min_sdf_z[:, None]

        return sample_z_vals.detach(), sample_z_vals_inv.detach()

    def approximate_normal(self, pts, dirs, max_delta_z=0.01, min_delta_z=0.001):
        delta_z = 0
        steps = int(max_delta_z/min_delta_z)

        is_appr = torch.zeros_like(pts[:, 0])
        grad_all = torch.zeros_like(pts)

        is_to_appr = torch.ones_like(pts[:, 0]).bool()
        pts_to_appr = pts
        dirs_to_appr = dirs
        for i in range(steps):
            delta_z = min_delta_z
            rand_theta = torch.rand(4) * np.pi * 2
            rand_phi = torch.rand(4) * np.pi
            rand_pts = torch.zeros(4, 3)
            rand_pts[:, 0] = torch.cos(rand_phi) * torch.cos(rand_theta)
            rand_pts[:, 1] = torch.cos(rand_phi) * torch.sin(rand_theta)
            rand_pts[:, 2] = torch.sin(rand_phi)
            rand_pts_norm = torch.mm(dirs_to_appr, rand_pts.transpose(0, 1))
            rand_pts = rand_pts[None, ...] - rand_pts_norm[..., None] * dirs_to_appr[:, None, :]
            rand_pts = rand_pts * delta_z / 10
            rand_pts = torch.cat([rand_pts, torch.zeros_like(rand_pts[:, :1, :])], dim=1)
            rand_pts = rand_pts + pts_to_appr[:, None, :] + dirs_to_appr[:, None, :] * delta_z

            rand_pts_grad = self.sdf_network.gradient(rand_pts.reshape(-1, 3)).reshape(-1, 5, 3)
            grad_mean = rand_pts_grad.mean(dim=1)
            grad_norm = torch.linalg.norm(grad_mean, dim=-1)
            grad_var = rand_pts_grad.var(dim=1).sum(-1)

            if i == steps - 1:
                grad_all[is_to_appr, :] = grad_mean
            else:
                is_norm_reliable = (grad_norm > 0.5) & (grad_var < 0.1)
                grad_all[is_to_appr, :][is_norm_reliable, :] = grad_mean[is_norm_reliable, :]
                is_appr[is_to_appr][is_norm_reliable] = 1.

            is_to_appr = (is_appr < 1)
            pts_to_appr = pts[is_to_appr, :]
            dirs_to_appr = dirs[is_to_appr, :]

        return grad_all








