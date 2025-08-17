import torch
import math
from einops import rearrange
from models.pointops.functions import pointops
import logging
import os
import numpy as np
import random
from torch.autograd import grad
from einops import rearrange, repeat
from sklearn.neighbors import NearestNeighbors
from models.Chamfer3D.dist_chamfer_3D import chamfer_3DDist
chamfer_dist = chamfer_3DDist()
from models import torch_tensor_functions


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def index_points(pts, idx):
    """
    Input:
        pts: input points data, [B, C, N]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, C, S, [K]]
    """
    batch_size = idx.shape[0]
    sample_num = idx.shape[1]
    fdim = pts.shape[1]
    reshape = False
    if len(idx.shape) == 3:
        reshape = True
        idx = idx.reshape(batch_size, -1)
    # (b, c, (s k))
    res = torch.gather(pts, 2, idx[:, None].repeat(1, fdim, 1))
    if reshape:
        res = rearrange(res, 'b c (s k) -> b c s k', s=sample_num)

    return res


def FPS(pts, fps_pts_num):
    # input: (b, 3, n)

    # (b, n, 3)
    pts_trans = rearrange(pts, 'b c n -> b n c').contiguous()
    # (b, fps_pts_num)
    sample_idx = pointops.furthestsampling(pts_trans, fps_pts_num).long()
    # (b, 3, fps_pts_num)
    sample_pts = index_points(pts, sample_idx)

    return sample_pts


def get_knn_pts(k, pts, center_pts, return_idx=False):
    # input: (b, 3, n)

    # (b, n, 3)
    pts_trans = rearrange(pts, 'b c n -> b n c').contiguous()
    # (b, m, 3)
    center_pts_trans = rearrange(center_pts, 'b c m -> b m c').contiguous()
    # (b, m, k)
    knn_idx = pointops.knnquery_heap(k, pts_trans, center_pts_trans).long()
    # (b, 3, m, k)
    knn_pts = index_points(pts, knn_idx)

    if return_idx == False:
        return knn_pts
    else:
        return knn_pts, knn_idx


def midpoint_interpolate(args, sparse_pts):
    # sparse_pts: (b, 3, n)

    pts_num = sparse_pts.shape[-1]
    up_pts_num = int(pts_num * args.upsampling_rate)
    k = int(2 * args.upsampling_rate)
    # (b, 3, n, k)
    knn_pts = get_knn_pts(k, sparse_pts, sparse_pts)
    # (b, 3, n, k)
    repeat_pts = repeat(sparse_pts, 'b c n -> b c n k', k=k)
    # (b, 3, n, k)
    mid_pts = (knn_pts + repeat_pts) / 2.0
    # (b, 3, (n k))
    mid_pts = rearrange(mid_pts, 'b c n k -> b c (n k)')
    # note that interpolated_pts already contain sparse_pts
    interpolated_pts = mid_pts
    # fps: (b, 3, up_pts_num)
    interpolated_pts = FPS(interpolated_pts, up_pts_num)

    return interpolated_pts

def normalize_point_cloud(input, centroid=None, furthest_distance=None):
    # input: (b, 3, n) tensor

    if centroid is None:
        # (b, 3, 1)
        centroid = torch.mean(input, dim=-1, keepdim=True)
    # (b, 3, n)
    input = input - centroid
    if furthest_distance is None:
        # (b, 3, n) -> (b, 1, n) -> (b, 1, 1)
        furthest_distance = torch.max(torch.norm(input, p=2, dim=1, keepdim=True), dim=-1, keepdim=True)[0]
    input = input / furthest_distance

    return input, centroid, furthest_distance

def add_noise(pts, sigma, clamp):
    # input: (b, 3, n)

    assert (clamp > 0)
    jittered_data = torch.clamp(sigma * torch.randn_like(pts), -1 * clamp, clamp).cuda()
    jittered_data += pts

    return jittered_data

# generate patch for test
def extract_knn_patch(k, pts, center_pts):
    # input : (b, 3, n)

    # (n, 3)
    pts_trans = rearrange(pts.squeeze(0), 'c n -> n c').contiguous()
    pts_np = pts_trans.detach().cpu().numpy()
    # (m, 3)
    center_pts_trans = rearrange(center_pts.squeeze(0), 'c m -> m c').contiguous()
    center_pts_np = center_pts_trans.detach().cpu().numpy()
    knn_search = NearestNeighbors(n_neighbors=k, algorithm='auto')
    knn_search.fit(pts_np)
    # (m, k)
    knn_idx = knn_search.kneighbors(center_pts_np, return_distance=False)
    # (m, k, 3)
    patches = np.take(pts_np, knn_idx, axis=0)
    patches = torch.from_numpy(patches).float().cuda()
    # (m, 3, k)
    patches = rearrange(patches, 'm k c -> m c k').contiguous()

    return patches

def get_logger(name, log_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')
    # output to console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    # output to log file
    log_name = name + '_log.txt'
    file_handler = logging.FileHandler(os.path.join(log_dir, log_name))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def reset_model_args(train_args, model_args):
    for arg in vars(train_args):
        setattr(model_args, arg, getattr(train_args, arg))
        
def chamfer_sqrt(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    d1 = torch.clamp(d1, min=1e-9)
    d2 = torch.clamp(d2, min=1e-9)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    return (d1 + d2) / 2

def get_cd_loss(args, coarse_pts, gt_pts):
    loss_cd = chamfer_sqrt(coarse_pts.permute(0,2,1).contiguous(), gt_pts.permute(0,2,1).contiguous()) * 1e3
    return loss_cd

def loss_on_proj(p0, p1):
  # p0 : B, M, 3
  # p1 : B, N, 3
  thisbatchsize = p0.size()[0]
  output = 0
  dis_map = torch_tensor_functions.compute_sqrdis_map(p0, p1)  # B, M, N

  neighbour_id_01 = torch.topk(dis_map, k=5, dim=-1, largest=False)[1]
  neighbour_dis_01 = torch.topk(dis_map, k=5, dim=-1, largest=False)[0]
  neighbour_id_01 = neighbour_id_01[:, :, 1:]
  neighbour_coor_01 = torch_tensor_functions.indexing_neighbor(p1, neighbour_id_01)
  neighbour_dis_01 = neighbour_dis_01[:, :, 1:]
  neighbour_weight_01 = neighbour_dis_01.detach() * 1000
  neighbour_weight_01 = torch.exp(-1 * neighbour_weight_01)
  neighbour_weight_01 = neighbour_weight_01 / (torch.sum(neighbour_weight_01, dim=-1, keepdim=True) + 0.00001)
  dis_01 = p0.view(thisbatchsize, -1, 1, 3) - neighbour_coor_01
  dis_01 = torch.sum(torch.mul(dis_01, dis_01), dim=-1, keepdim=False)
  pro_dis_01 = torch.mul(neighbour_weight_01, dis_01)
  output += 0.5 * torch.sum(pro_dis_01)

  neighbour_id_10 = torch.topk(dis_map, k=5, dim=1, largest=False)[1].transpose(2, 1)
  neighbour_dis_10 = torch.topk(dis_map, k=5, dim=1, largest=False)[0].transpose(2, 1)
  neighbour_id_10 = neighbour_id_10[:, :, 1:]
  neighbour_coor_10 = torch_tensor_functions.indexing_neighbor(p0, neighbour_id_10)
  neighbour_dis_10 = neighbour_dis_10[:, :, 1:]
  neighbour_weight_10 = neighbour_dis_10.detach() * 1000
  neighbour_weight_10 = torch.exp(-1 * neighbour_weight_10)
  neighbour_weight_10 = neighbour_weight_10 / (torch.sum(neighbour_weight_10, dim=-1, keepdim=True) + 0.00001)
  dis_10 = p1.view(thisbatchsize, -1, 1, 3) - neighbour_coor_10
  dis_10 = torch.sum(torch.mul(dis_10, dis_10), dim=-1, keepdim=False)
  pro_dis_10 = torch.mul(neighbour_weight_10, dis_10)
  output += 0.5 * torch.sum(pro_dis_10)

  return output / thisbatchsize