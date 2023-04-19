import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.embedder import get_embedder


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False,
                 mode='act'):
        super(SDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

        self.mode = mode

    def forward(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        x = self.activation(x)

        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x):
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class RenderingNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires_view=0,
                 squeeze_out=True):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

    def forward(self, points, normals, view_dirs, feature_vectors, sdf=None, second_order_gradients=None):
        view_dirs_raw = view_dirs
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = None

        if self.mode == 'idr' or self.mode == 'normal_appr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)
        elif self.mode == 'udf':
            rendering_input = torch.cat([points, view_dirs, sdf, feature_vectors], dim=-1)
        elif self.mode == 'second_order_udf':
            rendering_input = torch.cat([points, view_dirs, second_order_gradients, feature_vectors], dim=-1)
        elif self.mode == 'approximate_udf':
            appr_gradients = (normals[:, 2:] - normals[:, :-2])
            appr_zeros = torch.zeros_like(appr_gradients[:, :1])
            appr_gradients = torch.cat([appr_zeros, appr_gradients, appr_zeros], dim=-1)
            rendering_input = torch.cat([points, view_dirs, appr_gradients, feature_vectors], dim=-1)
        elif self.mode == 'multi_udf':
            prev_normals = torch.cat([torch.zeros_like(normals[:, :1]), normals[:, :-1]], dim=-1)
            next_normals = torch.cat([normals[:, 1:], torch.zeros_like(normals[:, :1])], dim=-1)
            rendering_input = torch.cat([points, view_dirs, prev_normals, normals, next_normals, feature_vectors], dim=-1)
        elif self.mode == 'positive_normal_z':
            normals_z = normals[..., 2]
            is_invert = torch.where(normals_z>0, torch.ones_like(normals_z), -torch.ones_like(normals_z))
            rendering_input = torch.cat([points, view_dirs, normals*is_invert[..., None], feature_vectors], dim=-1)
        elif self.mode == 'prev_normal':
            prev_normals = torch.cat([torch.zeros_like(normals[:, :1]), normals[:, :-1]], dim=-1)
            rendering_input = torch.cat([points, view_dirs, prev_normals, feature_vectors], dim=-1)
        elif self.mode == 'random_normal':
            sign = torch.sign(torch.randn(1)).item()
            rendering_input = torch.cat([points, view_dirs, normals*sign, feature_vectors], dim=-1)
        elif self.mode == 'npnz':
            n_normals = normals / torch.linalg.norm(normals, ord=2, dim=-1)[..., None]
            normals_z = n_normals[..., 2]
            is_invert = torch.where(normals_z>0, torch.ones_like(normals_z), -torch.ones_like(normals_z))
            rendering_input = torch.cat([points, view_dirs, n_normals*is_invert[..., None], feature_vectors], dim=-1)
        elif self.mode == 'multi_npnz':
            n_normals = normals / torch.linalg.norm(normals, ord=2, dim=-1)[..., None]
            normals_z = n_normals[..., 2]
            is_invert_z = torch.where(normals_z > 0, torch.ones_like(normals_z), -torch.ones_like(normals_z))
            normals_y = n_normals[..., 1]
            is_invert_y = torch.where(normals_y > 0, torch.ones_like(normals_y), -torch.ones_like(normals_y))
            normals_x = n_normals[..., 0]
            is_invert_x = torch.where(normals_x > 0, torch.ones_like(normals_x), -torch.ones_like(normals_x))
            rendering_input = torch.cat([points, view_dirs,
                                         n_normals * is_invert_z[..., None],
                                         n_normals * is_invert_y[..., None],
                                         n_normals * is_invert_x[..., None],
                                         feature_vectors], dim=-1)
        elif self.mode == 'angle':
            norm = torch.linalg.norm(normals, ord=2, dim=-1, keepdim=True)
            theta = torch.acos(normals[:, :1]/((normals[:, 1:2]**2+normals[:, :1]**2)**0.5+1e-10))*torch.sign(normals[:, 1:2])
            phi = torch.asin(normals[:, -1:]/(norm+1e-10))
            rendering_input = torch.cat([points, view_dirs, norm, theta, phi, feature_vectors], dim=-1)
        elif self.mode == 'reflect':
            normals = normals / torch.linalg.norm(normals)
            # print(view_dirs_raw.shape)
            # print(normals.shape)
            view_cos = (view_dirs_raw * normals).sum(dim=-1, keepdim=True)
            reflect_dir = view_dirs_raw - 2 * view_cos * normals
            rendering_input = torch.cat([points, view_dirs, reflect_dir, feature_vectors], dim=-1)
        elif self.mode == 'obtuse_normal':
            cos = (normals * view_dirs_raw).sum(dim=-1, keepdim=True)
            is_invert = torch.where(cos<0, torch.ones_like(cos), -torch.ones_like(cos))
            rendering_input = torch.cat([points, view_dirs, normals*is_invert, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x


# This implementation is borrowed from nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch
class NeRF(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 d_in=3,
                 d_in_view=3,
                 multires=0,
                 multires_view=0,
                 output_ch=4,
                 skips=[4],
                 use_viewdirs=False):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=d_in_view)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view

        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)])

        ### Implementation according to the official code release
        ### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            return alpha, rgb
        else:
            assert False


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1]) * torch.exp(self.variance * 10.0)
