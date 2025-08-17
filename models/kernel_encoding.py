import torch
import torch.nn as nn
import math
from torch import Tensor
from torch.nn.init import kaiming_uniform_
import numpy as np

from kernels.kernel_points import load_kernels
from models.generic_blocks import gather, index_select, radius_gaussian, local_maxpool, UnaryBlock, NormBlock, DropPathPack, build_mlp, mlp_from_list

# Deformable KPConvX
class DeformableKPConvX(nn.Module):
    def __init__(self, args, need_query_features = False):
        super(DeformableKPConvX, self).__init__()
        self.conv_radius = args.conv_radius
        self.kernel_radius = args.kernel_radius
        self.kernel_shells_receptive_radius = torch.tensor(args.kernel_shells_receptive_radius)
        self.neighbor_limits = args.neighbor_limits
        self.shell_sizes = args.shell_sizes
        self.num_kernel_points = np.sum(self.shell_sizes).item()
        self.inf = 1e6
        self.influence_mode = args.influence_mode
        self.dimension = 3
        self.fixed_kernel_points = args.fixed_kernel_points
        self.kernel_output_dim = args.kernel_output_dim
        self.feature_dim = args.feature_dim
        self.attention_groups = args.groups
        self.need_query_features = need_query_features
        self.aggregation_mode = args.kp_aggregation
        
        in_channels_per_group = self.feature_dim // ((self.attention_groups + 1) // 2)
        out_channels_per_group = self.kernel_output_dim // ((self.attention_groups + 1) // 2)
        self.in_channels_per_group = in_channels_per_group
        self.out_channels_per_group = out_channels_per_group
        self.channel_offsets = in_channels_per_group // 2
        
        self.middle_dim = self.attention_groups * in_channels_per_group
        
        # Running variable containing deformed KP distance to input points. (used in regularization loss)
        self.min_d2 = None
        self.deformed_KP = None
        self.offset_features = None
        
        # Initialize weights
        if self.attention_groups == 1:
            weights = torch.zeros(size=(self.num_kernel_points, self.feature_dim, self.kernel_output_dim))
            deformed_weights = torch.zeros(size=(self.num_kernel_points, self.feature_dim, self.kernel_output_dim))
        else:
            weights = torch.zeros(size=(self.num_kernel_points, self.attention_groups, in_channels_per_group, out_channels_per_group))
            deformed_weights = torch.zeros(size=(self.num_kernel_points, self.attention_groups, in_channels_per_group, out_channels_per_group))
        
        self.weights = nn.Parameter(weights, requires_grad=True)
        self.deformed_weights = nn.Parameter(deformed_weights, requires_grad=True)
        
        self.deforming_begin_conv = nn.Conv1d(self.feature_dim, self.kernel_output_dim, 1)
        self.deforming_act = nn.LeakyReLU(0.1)
        self.deforming_end_conv = nn.Conv1d(self.middle_dim, self.feature_dim, 1)
        self.deforming_out_act = nn.LeakyReLU(0.1)
        self.deforming_conv = nn.Conv1d(self.feature_dim, 3 * self.num_kernel_points, 1)
        
        self.begin_conv = nn.Conv1d(self.feature_dim, self.kernel_output_dim, 1)
        self.act = nn.LeakyReLU(0.1)
        self.end_conv = nn.Conv1d(self.middle_dim, self.kernel_output_dim, 1)
        self.out_act = nn.LeakyReLU(0.1)
        
        self.offset_dim = self.dimension * self.num_kernel_points
        self.offset_bias = nn.Parameter(torch.zeros(self.offset_dim, dtype=torch.float32), requires_grad=True)
        
        if self.need_query_features:
            self.query_mlp = nn.Sequential(
                nn.Conv2d(self.kernel_output_dim + self.middle_dim, self.kernel_output_dim + self.middle_dim, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.kernel_output_dim + self.middle_dim, args.trans_dim, 1),
                nn.ReLU(inplace=True)
            )
        else:
            self.query_mlp = None
        
        # Reset parameters
        self.reset_parameters()

        # Initialize kernel points
        kernel_points = self.initialize_kernel_points()
        self.register_buffer("kernel_points", kernel_points)
        
    def reset_parameters(self):
        kaiming_uniform_(self.weights, a=math.sqrt(5))
        kaiming_uniform_(self.deformed_weights, a=math.sqrt(5))
        return

    def initialize_kernel_points(self) -> Tensor:
        """
        Initialize the kernel point positions in a sphere
        :return: the tensor of kernel points
        """
        kernel_points = load_kernels(self.kernel_radius, self.shell_sizes, dimension=self.dimension, fixed=self.fixed_kernel_points)
        return torch.from_numpy(kernel_points).float()
    
    def square_distance(self, src, dst):
        """
        Calculate Euclid distance between each two points.
        src^T * dst = xn * xm + yn * ym + zn * zm;
        sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
        sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
        dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
            = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
        Input:
            src: source points, [B, N, C]
            dst: target points, [B, M, C]
        Output:
            dist: per-point square distance, [B, N, M]
        """
        B, N, _ = src.shape
        _, M, _ = dst.shape
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
        dist += torch.sum(src ** 2, -1).view(B, N, 1)
        dist += torch.sum(dst ** 2, -1).view(B, 1, M)
        return dist
    
    def query_ball_point(self, xyz, new_xyz, radius, neighbor_limits):
        """
        Input:
            radius: local region radius
            nsample: max sample number in local region
            xyz: all points, [B, N, 3]
            new_xyz: query points, [B, S, 3]
        Return:
            group_idx: grouped points index, [B, S, nsample]
        """
        device = xyz.device
        B, N, C = xyz.shape
        _, S, _ = new_xyz.shape
        group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
        sqrdists = self.square_distance(new_xyz, xyz)
        group_idx[sqrdists > radius ** 2] = N
        group_idx = group_idx.sort(dim=-1)[0][:, :, :neighbor_limits]
        return group_idx
    
    def index_points(self, points, idx, pad_inf=False):
        """
        Input:
            points: input points data, [B, N, C]
            idx: sample index data, [B, S]
        Return:
            new_points:, indexed points data, [B, S, C]
        """
        points = points.permute(0,2,1).contiguous()
        B, N, C = points.shape
        device = points.device
        if pad_inf:
            shadow_points= torch.zeros(B, 1, C) + self.inf 
        else:
            shadow_points= torch.zeros(B, 1, C)
        shadow_points = shadow_points.to(device)
        cat_points = torch.cat([points, shadow_points], dim=1) # B, N+1, C

        B = cat_points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = cat_points[batch_indices, idx, :]
        return new_points
    
    def get_channel_shifted_features(self, features):
        # (B, N, K, C) -> (B, N, K, G, C//G)
        channel_shifted_features = []
        
        i = 0
        for m in range(self.attention_groups):
            i += 1
            features_m = features[:, :, :, m * self.channel_offsets : m * self.channel_offsets + self.in_channels_per_group]
            channel_shifted_features.append(features_m)
            
        channel_shifted_features = torch.stack(channel_shifted_features, dim=3)
        
        return channel_shifted_features
    
    def get_deformed_points(self, support_points, support_features):
        B, _, N = support_points.shape
        flipped_support_points = support_points.permute(0, 2, 1) # B, N, D
        neighbor_idx = self.query_ball_point(flipped_support_points, flipped_support_points, self.conv_radius, self.neighbor_limits) # (B, N, H)
        
        # kpconvd
        neighbor_pos = self.index_points(support_points, neighbor_idx, pad_inf=True).permute(0,3,1,2).contiguous() # (B, D, N, H)
        deforming_begin_features = self.deforming_begin_conv(support_features) # (B, C, N)
        neighbor_features = self.index_points(deforming_begin_features, neighbor_idx, pad_inf=False) # (B, N, H, C)
        neighbor_pos = neighbor_pos - support_points.unsqueeze(-1)  # (B, D, N, H) - (B, D, N, 1) = (B, D, N, H)
        
        # Get nearest kernel point (N, H) and weights applied to each neighbors (N, H)
        # Get Kernel point distances to neigbors
        differences = neighbor_pos.unsqueeze(-1) - self.kernel_points.view(1, 3, 1, 1, self.num_kernel_points) # (B, D, N, H, 1) - (1, D, 1, 1, K) = (B, D, N, H, K)
        sq_distances = torch.sum(differences ** 2, dim=1)  # (B, N, H, K)
        
        # # Get nearest kernel point (N, H), values < K
        # # In case of nearest mode, only the nearest KP can influence each point
        # nn_sq_dists, neighbors_1nn = torch.min(sq_distances, dim=3)
        influence_weights = None
        
        if self.influence_mode != 'constant':
            # Get Kernel point influences
            if self.influence_mode == 'linear':
                # Influence decrease linearly with the distance, and get to zero when d = sigma.
                influence_weights = torch.clamp(1 - torch.sqrt(sq_distances) / self.kernel_shells_receptive_radius, min=0.0)  # (B, M, H, K)
            elif self.influence_mode == 'gaussian':
                # Influence in gaussian of the distance.
                gaussian_sigma = self.kernel_shells_receptive_radius * 0.3
                influence_weights = radius_gaussian(sq_distances, gaussian_sigma)  # (B, M, H, K)
            else:
                raise ValueError("Unknown influence mode: : '{:s}'.  Should be 'constant', 'linear', or 'gaussian'".format(self.influence_mode))
            
        influence_weights = torch.transpose(influence_weights, 2, 3)  # (B, N, H, K) -> (B, N, K, H)
        # In case of nearest mode, only the nearest KP can influence each point
        if self.aggregation_mode == 'nearest':
            neighbors_1nn = torch.argmin(sq_distances.detach(), dim=3)
            influence_weights *= torch.transpose(nn.functional.one_hot(neighbors_1nn, self.num_kernel_points),  2, 3)

        elif self.aggregation_mode != 'sum':
            raise ValueError("Unknown aggregation mode: '{:s}'. Should be 'nearest' or 'sum'".format(self.aggregation_mode))
            
        # Apply distance weights
        deforming_weighted_feats = torch.matmul(influence_weights, neighbor_features)  # (B, N, K, H) x (B, N, H, C) -> (B, N, K, C)
        # group conv
        # deforming_weighted_feats = deforming_weighted_feats.view(B, -1, self.num_kernel_points, self.attention_groups, self.in_channels_per_group)  # (B, N, K, C) -> (B, N, K, G, C//G)
        deforming_weighted_feats = self.get_channel_shifted_features(deforming_weighted_feats)
        offset_features = torch.einsum("bmkgc,kgcd->bmgd", deforming_weighted_feats, self.deformed_weights)  # (B, N, K, G, C//G) * (B, K, G, C//G, O//G) -> (B, N, G, C//G)
        offset_features = offset_features.reshape((B, -1, self.middle_dim))  # (B, N, G, C//G) -> (B, N, C)
        offset_features = offset_features.permute(0,2,1).contiguous()
        offset_features = self.deforming_act(offset_features)
        offset_features = self.deforming_end_conv(offset_features)
        offset_features = self.deforming_out_act(offset_features)
        unscaled_offsets = torch.tanh(self.deforming_conv(offset_features))
        unscaled_offsets = unscaled_offsets.view(B, 3, -1, N).permute(0, 1, 3, 2).contiguous() # (B, 3, N, K)
        offsets = unscaled_offsets * self.conv_radius # (B, 3, N, K)
        
        return offsets, neighbor_pos, neighbor_idx
    
    def forward(self, support_points, support_features):
        B, _, N = support_points.shape
        offsets, neighbor_pos, neighbor_idx = self.get_deformed_points(support_points, support_features) # (B, 3, N, K) (B, 3, N, H) (B, N, H)
        deformed_ks_points = self.kernel_points.view(1, 3, 1, self.num_kernel_points) + offsets # (B, 3, N, K)
        begin_features = self.begin_conv(support_features) # (B, C, N)
        neighbor_features = self.index_points(begin_features, neighbor_idx, pad_inf = False) # (B, N, H, C)
        differences = neighbor_pos.unsqueeze(-1) - deformed_ks_points.unsqueeze(-2) # (B, 3, N, H, 1) - (B, 3, N, 1, K) = (B, 3, N, H, K)
        sq_distances = torch.sum(differences ** 2, dim=1)  # (B, N, H, K)
        
        # # Save min distances for loss
        # self.min_d2, _ = torch.min(sq_distances, dim=3)   # (B, N, H)
        with torch.no_grad():

            # Get Kernel point influences
            if self.influence_mode == 'constant':
                # Every point get an influence of 1.
                influence_weights = torch.ones_like(sq_distances.detach())

            elif self.influence_mode == 'linear':
                # Influence decrease linearly with the distance, and get to zero when d = sigma.
                influence_weights = torch.clamp(1 - torch.sqrt(sq_distances.detach()) / self.kernel_shells_receptive_radius, min=0.0)  # (B, N, H, K)

            elif self.influence_mode == 'gaussian':
                # Influence in gaussian of the distance.
                gaussian_sigma = self.sigma * 0.3
                influence_weights = radius_gaussian(sq_distances.detach(), gaussian_sigma)
            else:
                raise ValueError("Unknown influence mode: : '{:s}'.  Should be 'constant', 'linear', or 'gaussian'".format(self.aggregation_mode))
            influence_weights = torch.transpose(influence_weights, 2, 3)  # (B, N, H, K) -> (B, N, K, H)

            # In case of nearest mode, only the nearest KP can influence each point
            if self.aggregation_mode == 'nearest':
                neighbors_1nn = torch.argmin(sq_distances.detach(), dim=3)
                influence_weights *= torch.transpose(nn.functional.one_hot(neighbors_1nn, self.num_kernel_points),  2, 3)

            elif self.aggregation_mode != 'sum':
                raise ValueError("Unknown aggregation mode: '{:s}'. Should be 'nearest' or 'sum'".format(self.aggregation_mode))
            
        # Apply distance weights
        weighted_feats = torch.matmul(influence_weights, neighbor_features)  # (B, N, K, H) x (B, N, H, C) -> (B, N, K, C)
        # group conv
        # weighted_feats = weighted_feats.view(B, -1, self.num_kernel_points, self.attention_groups, self.in_channels_per_group)  # (B, N, K, C) -> (B, N, K, G, C//G)
        weighted_feats = self.get_channel_shifted_features(weighted_feats)
        output_features = torch.einsum("bmkgc,kgcd->bmgd", weighted_feats, self.weights)  # (B, N, K, G, C//G) * (B, K, G, C//G, O//G) -> (B, N, G, C//G)
        output_features = output_features.reshape((B, -1, self.middle_dim))  # (B, N, G, C//G) -> (B, N, C)
        output_features = output_features.permute(0,2,1).contiguous()
        output_features = self.act(output_features)
        output_features = self.end_conv(output_features)
        output_features = self.out_act(output_features + support_features)
        
        if self.query_mlp != None:
            weighted_feats = weighted_feats.reshape(B, N, -1, self.middle_dim)
            fuse_feats = [weighted_feats.permute(0, 3, 1, 2).contiguous(), output_features.unsqueeze(-1).repeat(1,1,1,self.num_kernel_points)] #  (B, d*2, n, nkp)
            fuse_feats = self.query_mlp(torch.cat(fuse_feats, dim=1)) #  (B, d, n, nkp)
        else:
            fuse_feats = None
        
        return output_features, [deformed_ks_points, fuse_feats]

class KernelEncoder(nn.Module):
    def __init__(self, args):
        super(KernelEncoder, self).__init__()
        self.kernel_shells_encoder_1 = DeformableKPConvX(args)
        self.kernel_shells_encoder_2 = DeformableKPConvX(args)
        self.kernel_shells_encoder_3 = DeformableKPConvX(args, need_query_features = True)

    def forward(self, pts, features):
        encoding_features1, _ = self.kernel_shells_encoder_1(pts, features)
        encoding_features2, _ = self.kernel_shells_encoder_2(pts, encoding_features1)
        encoding_features3, [deformed_ks_points, query_features] = self.kernel_shells_encoder_3(pts, encoding_features2)

        return encoding_features3, deformed_ks_points, query_features

class KernelDecoder(nn.Module):
    def __init__(self, args):
        super(KernelDecoder, self).__init__()
        self.conv_radius = args.conv_radius
        self.kernel_radius = args.kernel_radius
        self.kernel_shells_receptive_radius = torch.tensor(args.kernel_shells_receptive_radius)
        self.neighbor_limits = args.neighbor_limits
        self.inf = 1e6
        self.influence_mode = args.influence_mode
        self.aggregation_mode = args.kp_aggregation
        self.dimension = 3
        self.fixed_kernel_points = args.fixed_kernel_points
        self.kernel_output_dim = args.kernel_output_dim
        self.feature_dim = args.feature_dim
        self.attention_groups = args.groups
        
        self.begin_conv = nn.Conv1d(self.feature_dim, self.kernel_output_dim, 1)
        self.act = nn.LeakyReLU(0.1)
        self.end_conv = nn.Conv1d(self.kernel_output_dim, self.kernel_output_dim, 1)
        self.out_act = nn.LeakyReLU(0.1)
        
        self.query_mlp = nn.Sequential(
                nn.Conv2d(self.kernel_output_dim * 2 + 3, self.kernel_output_dim * 2, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.kernel_output_dim * 2, args.trans_dim, 1),
                nn.ReLU(inplace=True)
            )

    def initialize_kernel_points(self, shell_sizes) -> Tensor:
        """
        Initialize the kernel point positions in a sphere
        :return: the tensor of kernel points
        """
        kernel_points = load_kernels(self.kernel_radius, shell_sizes, dimension=self.dimension, fixed=self.fixed_kernel_points)
        return torch.from_numpy(kernel_points).float()
    
    def square_distance(self, src, dst):
        """
        Calculate Euclid distance between each two points.
        src^T * dst = xn * xm + yn * ym + zn * zm;
        sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
        sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
        dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
            = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
        Input:
            src: source points, [B, N, C]
            dst: target points, [B, M, C]
        Output:
            dist: per-point square distance, [B, N, M]
        """
        B, N, _ = src.shape
        _, M, _ = dst.shape
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
        dist += torch.sum(src ** 2, -1).view(B, N, 1)
        dist += torch.sum(dst ** 2, -1).view(B, 1, M)
        return dist
    
    def query_ball_point(self, xyz, new_xyz, radius, neighbor_limits):
        """
        Input:
            radius: local region radius
            nsample: max sample number in local region
            xyz: all points, [B, N, 3]
            new_xyz: query points, [B, S, 3]
        Return:
            group_idx: grouped points index, [B, S, nsample]
        """
        device = xyz.device
        B, N, C = xyz.shape
        _, S, _ = new_xyz.shape
        group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
        sqrdists = self.square_distance(new_xyz, xyz)
        group_idx[sqrdists > radius ** 2] = N
        group_idx = group_idx.sort(dim=-1)[0][:, :, :neighbor_limits]
        return group_idx
    
    def index_points(self, points, idx, pad_inf=False):
        """
        Input:
            points: input points data, [B, N, C]
            idx: sample index data, [B, S]
        Return:
            new_points:, indexed points data, [B, S, C]
        """
        points = points.permute(0,2,1).contiguous()
        B, N, C = points.shape
        device = points.device
        if pad_inf:
            shadow_points= torch.zeros(B, 1, C) + self.inf 
        else:
            shadow_points= torch.zeros(B, 1, C)
        shadow_points = shadow_points.to(device)
        cat_points = torch.cat([points, shadow_points], dim=1) # B, N+1, C

        B = cat_points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = cat_points[batch_indices, idx, :]
        return new_points
    
    def forward(self, support_points, support_features, upsampling_size):
        B, _, N = support_points.shape
        # Initialize kernel points
        # shell_sizes = [1, upsampling_size]
        first_shell_size = round(upsampling_size / 3)
        shell_sizes = [1, upsampling_size - first_shell_size, first_shell_size]
        num_kernel_points = np.sum(shell_sizes).item()
        query_points = self.initialize_kernel_points(shell_sizes).cuda()
        flipped_support_points = support_points.permute(0, 2, 1) # B, N, D
        neighbor_idx = self.query_ball_point(flipped_support_points, flipped_support_points, self.conv_radius, self.neighbor_limits) # (B, N, H)
        
        neighbor_positions = self.index_points(support_points, neighbor_idx, pad_inf = True).permute(0,3,1,2).contiguous() # (B, D, N, K)
        neighbor_positions = neighbor_positions - support_points.unsqueeze(-1) # (B, D, N, K) - (B, D, N, 1) = (B, D, N, K)
        differences = neighbor_positions.unsqueeze(-1) - query_points.view(1, 3, 1, 1, num_kernel_points) # (B, D, N, K, 1) - (1, D, 1, 1, H) = (B, D, N, K, H)
        sq_distances = torch.sum(differences ** 2, dim=1)  # (B, N, K, H)
        
        with torch.no_grad():

            # Get Kernel point influences
            if self.influence_mode == 'constant':
                # Every point get an influence of 1.
                influence_weights = torch.ones_like(sq_distances.detach())

            elif self.influence_mode == 'linear':
                # Influence decrease linearly with the distance, and get to zero when d = sigma.
                influence_weights = torch.clamp(1 - torch.sqrt(sq_distances.detach()) / self.kernel_shells_receptive_radius, min=0.0)  # (B, M, H, K)

            elif self.influence_mode == 'gaussian':
                # Influence in gaussian of the distance.
                gaussian_sigma = self.sigma * 0.3
                influence_weights = radius_gaussian(sq_distances.detach(), gaussian_sigma)
            else:
                raise ValueError("Unknown influence mode: : '{:s}'.  Should be 'constant', 'linear', or 'gaussian'".format(self.aggregation_mode))
            influence_weights = torch.transpose(influence_weights, 2, 3)  # (B, M, H, K) -> (B, M, K, H)

            # In case of nearest mode, only the nearest KP can influence each point
            if self.aggregation_mode == 'nearest':
                neighbors_1nn = torch.argmin(sq_distances.detach(), dim=3)
                influence_weights *= torch.transpose(nn.functional.one_hot(neighbors_1nn, num_kernel_points),  2, 3)

            elif self.aggregation_mode != 'sum':
                raise ValueError("Unknown aggregation mode: '{:s}'. Should be 'nearest' or 'sum'".format(self.aggregation_mode))
            
        # Apply distance weights
        pre_encoding_features = self.begin_conv(support_features)
        neighbor_features = self.index_points(pre_encoding_features, neighbor_idx, pad_inf = False) # (B, M, H, C)
        weighted_feats = torch.matmul(influence_weights, neighbor_features)  # (B, M, K, H) x (B, M, H, C) -> (B, M, K, C)

        queries = self.query_mlp(torch.cat([weighted_feats.permute(0, 3, 1, 2).contiguous()[:, :, :, 1:], \
            query_points.permute(1, 0).contiguous().view(1, 3, 1, num_kernel_points).repeat(B, 1, N, 1)[:, :, :, 1:], \
            support_features.unsqueeze(-1).repeat(1,1,1,num_kernel_points - 1)], dim=1))
        
        return queries