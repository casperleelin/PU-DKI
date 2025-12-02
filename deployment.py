import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import argparse
from models.model_DeepKernel import ModelDeepKernel
from args import parse_pu1k_args, parse_pugan_args
from args.utils import *
from glob import glob
import open3d as o3d
from einops import repeat
from models.helps import *
import time

def _normalize_point_cloud(pc):
    # b, n, 3
    centroid = torch.mean(pc, dim=1, keepdim = True) # b, 1, 3
    pc = pc - centroid # b, n, 3
    furthest_distance = torch.max(torch.sqrt(torch.sum(pc**2, dim=-1, keepdim=True)), dim=1, keepdim=True)[0] # b, 1, 1
    pc = pc / furthest_distance
    return pc

def chamfer_sqrt(p1, p2):
    d1, d2, _, _ = chamfer_dist(_normalize_point_cloud(p1), _normalize_point_cloud(p2))
    d1 = torch.mean(d1)
    d2 = torch.mean(d2)
    return (d1 + d2)

def upsampling(args, model, input_pcd):
    pcd_pts_num = input_pcd.shape[-1]
    patch_pts_num = args.num_points
    sample_num = int(pcd_pts_num / patch_pts_num * args.patch_rate)
    seed = FPS(input_pcd, sample_num)
    patches = extract_knn_patch(patch_pts_num, input_pcd, seed)
    patches, centroid, furthest_distance = normalize_point_cloud(patches)
    
    all_coarse_pts = []
    batch_size = 250
    start = time.time()
    for i in range(0, len(patches), batch_size):
        batch_patches = patches[i : i + batch_size]
        batch_centroid = centroid[i : i + batch_size]
        batch_furthest_distance = furthest_distance[i : i + batch_size]
        coarse_pts, uniform_loss, repulsion_loss = model.forward(batch_patches)
        coarse_pts = batch_centroid + coarse_pts * batch_furthest_distance
        coarse_pts = rearrange(coarse_pts, 'b c n -> c (b n)').contiguous()
        # coarse_pts = FPS(coarse_pts.unsqueeze(0), coarse_pts.shape[-1])
        all_coarse_pts.append(coarse_pts)
        del batch_patches, coarse_pts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    final_coarse_pts = torch.cat(all_coarse_pts, dim=1)
    elapsed = time.time() - start
    print(f'interpolation time: {elapsed}s')

    fps_start = time.time()
    refined_pts = reliable_FPS(final_coarse_pts.unsqueeze(0), input_pcd.shape[-1]* args.upsampling_rate)
    fps_elapsed = time.time() - fps_start
    print(f'fps time: {fps_elapsed}s')
    return refined_pts

def upsampling_deployment(model, args):
    total_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("===number of trainable parameters in upsampler: {:.4f} K === ".format(float(total_trainable_parameters / 1e3)))
    with torch.no_grad():
        model.eval()
        test_input_path = glob(os.path.join(args.input_dir, '*.xyz'))
        for i, path in enumerate(test_input_path):
            print(f'This is the {path}')
            pcd = o3d.io.read_point_cloud(path)
            pcd_name = path.split('/')[-1]
            input_pcd = np.array(pcd.points)
            input_pcd = torch.from_numpy(input_pcd).float().cuda()
            input_pcd = rearrange(input_pcd, 'n c -> c n').contiguous()
            input_pcd = input_pcd.unsqueeze(0)

            input_pcd, centroid, furthest_distance = normalize_point_cloud(input_pcd)
            pcd_upsampled = upsampling(args, model, input_pcd)
            pcd_upsampled = centroid + pcd_upsampled * furthest_distance

            saved_pcd = rearrange(pcd_upsampled.squeeze(0), 'c n -> n c').contiguous()
            saved_pcd = saved_pcd.detach().cpu().numpy()
            save_folder = os.path.join(args.save_dir, 'xyz')
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            np.savetxt(os.path.join(save_folder, pcd_name), saved_pcd, fmt='%.6f')  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deployment Arguments')
    parser.add_argument('--input_dir', default='./output', type=str, help='path to folder of input point clouds')
    parser.add_argument('--save_dir', default='pcd', type=str, help='save upsampled point cloud and results')
    parser.add_argument('--ckpt', default='./output', type=str, help='checkpoints')
    parser.add_argument('--dataset', default='pugan', type=str, help='pu1k or pugan')
    parser.add_argument('--upsampling_rate', default=15, type=int, help='upsampling rate')
    parser.add_argument('--patch_num_ratio', type=float, default=3)
    args = parser.parse_args()

    if args.dataset == 'pugan':
        reset_model_args(parse_pugan_args(), args)
        model = ModelDeepKernel(args)
    else:
        reset_model_args(parse_pu1k_args(), args)
        model = ModelDeepKernel(args)
    print(args.ckpt)
    print(args.save_dir)
    model = model.cuda()
    model.load_state_dict(torch.load(args.ckpt))
    upsampling_deployment(model, args)