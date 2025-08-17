import torch
import torch.nn as nn
import math
from models.transformer import TransformerEncoder
from models.attention_queries import *
from models.attention_queries import CrossAttentionTransformer
from models.kernel_encoding import KernelEncoder, KernelDecoder
from models.pointops.functions import pointops
from models.residual_refiner import ResidualRefiner

# from https://github.com/yulequan/PU-Net/blob/dd768ea2eb349dc1a35d62f8c1fb019efc526fe7/code/model_utils.py
def get_repulsion_loss(pred, nsample=20, radius=0.07, h=0.03, min_num=(torch.zeros(1).cuda()+1e-9), device=None):
    min_num = min_num.to(device)
    # if knn:
    #     _, idx = knn_point_2(nsample, pred, pred)
    #     pts_cnt = tf.constant(nsample, shape=(30, 1024))
    # else:
    #     idx, pts_cnt = query_ball_point(radius, nsample, pred, pred)
    # tf.summary.histogram('smooth/unque_index', pts_cnt)
    # print(nsample)
    idx = pointops.ballquery(radius, nsample, pred, pred)  # (B, npoint, nsample)

    # grouped_pred = group_point(pred, idx)  # (batch_size, npoint, nsample, 3)
    grouped_pred = pointops.grouping(pred.transpose(1, 2).contiguous(), idx.detach()).permute(0, 2, 3,
                                                                                                   1).contiguous()  # return b,c,npoint,nsample, to b,npoint,nsample,c
    grouped_pred -= torch.unsqueeze(pred, 2)

    # get the uniform loss
    # if use_l1:
    #     dists = tf.reduce_sum(tf.abs(grouped_pred), axis=-1)
    # else:
    #     dists = tf.reduce_sum(grouped_pred ** 2, axis=-1)
    dists = torch.sum(grouped_pred ** 2, -1, keepdim=False, out=None)
    del grouped_pred, idx

    # val, idx = tf.nn.top_k(-dists, 5)
    dists = torch.topk(-dists, k=5)[0]
    dists = -dists[:, :, 1:]  # remove the first one
    # val = val[:, :, 1:]  # remove the first one

    # if use_l1:
    #     h = np.sqrt(h)*2
    # print(("h is ", h))

    # val = tf.maximum(0.0, h + val)  # dd/np.sqrt(n)
    # val = torch.max(torch.zeros_like(val), h + val)  # dd/np.sqrt(n)
    # dists = torch.nn.ReLU()(h + dists)
    # dists = torch.max(min_num, dists)
    dists = torch.nn.ReLU()(dists)
    dists_sqrt = torch.sqrt(dists)
    weight = torch.exp(-dists/h**2)
    del dists
    # repulsion_loss = torch.mean(val.view(-1), 0, keepdim=False)
    # return repulsion_loss
    return torch.mean((radius - dists_sqrt * weight).view(-1), 0, keepdim=False)

def knn_point(k, xyz1, xyz2):
    """
    Input:
        k: int32, number of k in k-nn search
        xyz1: (batch_size, ndataset, c) float32 array, input points
        xyz2: (batch_size, npoint, c) float32 array, query points
    Output:
        val: (batch_size, npoint, k) float32 array, L2 distances
        idx: (batch_size, npoint, k) int32 array, indices to input points
    """
    xyz1 = torch.unsqueeze(xyz1, 1)
    xyz2 = torch.unsqueeze(xyz2, 2)
    xyz1 = xyz1-xyz2
    del xyz2
    dist = torch.sum(xyz1 ** 2, -1)
    del xyz1
    val, idx = torch.topk(-dist, k=k)
    del dist

    return val, idx.int()
def get_uniform_loss(pcd, percentages=[0.004,0.008,0.010,0.012,0.016], radius=1.0):
    N = pcd.shape[1] 
    npoint = int(N * 0.05)
    loss=torch.zeros([1], requires_grad=True).cuda(non_blocking=True)
    for p in percentages:
        nsample = int(N*p)
        r = math.sqrt(p*radius)
        disk_area = math.pi *(radius ** 2) * p/nsample 
        new_xyz = pcd.transpose(1, 2).contiguous()  
        new_xyz = pointops.gathering(new_xyz, pointops.furthestsampling(pcd, npoint).detach())
        # new_xyz = utils.gather_operation(new_xyz, utils.furthest_point_sample(pcd, npoint).detach())
        new_xyz = new_xyz.transpose(1, 2).contiguous() 
        idx = pointops.ballquery(r, nsample, new_xyz, pcd) 
        # idx,pts_cnt = utils.query_ball_point(r, nsample, new_xyz, pcd) 
        del new_xyz
        expect_len =  math.sqrt(2*disk_area/1.732)
        grouped_pcd = pointops.grouping(pcd.transpose(1,2).contiguous(),idx.detach()).permute(0, 2, 3, 1).contiguous()
        # grouped_pcd = utils.grouping_operation(pcd.transpose(1,2).contiguous(),idx.detach()).permute(0, 2, 3, 1).contiguous() 
        grouped_pcd = torch.cat(torch.unbind(grouped_pcd,dim=1),0) 
        uniform_dis, _ = knn_point(2, grouped_pcd, grouped_pcd)
        uniform_dis = -uniform_dis[:, :, 1:]
        uniform_dis = torch.sqrt(torch.abs(uniform_dis+(1e-8))) 
        uniform_dis = torch.mean(uniform_dis,-1)
        uniform_dis = (uniform_dis - expect_len)**2 / (expect_len + 1e-8) 
        uniform_dis = uniform_dis.view(-1).contiguous() 
        loss+=torch.mean(uniform_dis,0)*math.pow(p*100,2)
        del uniform_dis, grouped_pcd, expect_len
    return loss/len(percentages) 


class ModelDeepKernel(nn.Module):
    def __init__(self, args):
        super(ModelDeepKernel, self).__init__()
        self.args = args
        self.local_features_dim = args.trans_dim
        self.upsampling_rate = args.upsampling_rate
        self.shell_sizes = args.shell_sizes
        self.num_kernel_points = np.sum(self.shell_sizes).item()
        self.feature_encoder = TransformerEncoder(args)
        
        self.kernel_encoder = KernelEncoder(args)
        self.kernel_decoder = KernelDecoder(args)

        self.local_geo_refiner = CrossAttentionTransformer(args)
        
        self.feature_fuser = nn.Conv2d(args.kernel_output_dim, args.kernel_output_dim, 1)
        
        self.residual_refiner = ResidualRefiner(num_neighbors=16, in_channels=args.kernel_output_dim)

        # MLPs for offset and scaling
        self.decoding_coords_mlp = nn.Sequential(
            nn.Conv1d(args.kernel_output_dim, 16, 1),  # Local + global
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 3, 1)
        )
    
    def forward(self, pts):
        B, _, N = pts.shape
        features = self.feature_encoder(pts)

        encoded_features, deformed_points, kernel_features = self.kernel_encoder(pts, features)
        local_geo_features = self.kernel_decoder(pts, encoded_features, self.upsampling_rate)

        local_refined_features = self.local_geo_refiner(local_geo_features, kernel_features)
        
        reshaped_local_features = local_refined_features.view(B, -1, N * self.upsampling_rate)

        local_offset_coords = self.decoding_coords_mlp(reshaped_local_features)
        coarse_coords = pts.unsqueeze(-1).repeat(1, 1, 1, self.upsampling_rate).view(pts.shape[0], 3, -1) + local_offset_coords
        global_repeated_features = features.unsqueeze(-1).repeat(1, 1, 1, self.upsampling_rate)
        pre_fusion_features = local_refined_features + global_repeated_features
        global_fused_features = self.feature_fuser(pre_fusion_features).view(B, -1, N * self.upsampling_rate)

        offset = self.residual_refiner(global_fused_features, coarse_coords)
        
        refined_coords = coarse_coords + offset

        if self.training:
            uniform_loss = get_uniform_loss(refined_coords.transpose(1, 2).contiguous())
            repulsion_loss = get_repulsion_loss(refined_coords.transpose(1, 2).contiguous(), device=pts.device)
        else:
            uniform_loss = 0
            repulsion_loss = 0

        return refined_coords, uniform_loss, repulsion_loss
        