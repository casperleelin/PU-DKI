import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
from cd.Chamfer3D.dist_chamfer_3D import chamfer_3DDist
chamfer_dist = chamfer_3DDist()
from geomloss import SamplesLoss
import open3d as o3d

emd_loss = SamplesLoss(loss="sinkhorn", p=1, blur=0.01, scaling=0.9, backend="tensorized")

def chamfer_sqrt(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    d1 = torch.clamp(d1, min=1e-9)
    d2 = torch.clamp(d2, min=1e-9)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    return (d1 + d2) / 2


source_file = 'real_rock_outcrop/Real/volcanic.xyz'
target_file = 'real_rock_outcrop/Real_up_PUCRN/volcanic.xyz'

pcd = o3d.io.read_point_cloud(source_file)
pcd_name = source_file.split('/')[-1]
gt = torch.Tensor(np.asarray(o3d.io.read_point_cloud(target_file).points)).unsqueeze(0).cuda()
input_pcd = np.array(pcd.points)
input_pcd = torch.from_numpy(input_pcd).float().cuda()
input_pcd = input_pcd.unsqueeze(0)
cd = chamfer_sqrt(input_pcd.contiguous(), gt).cpu().item()
# ed = emd_loss(input_pcd.cpu(), gt.cpu()).cpu().item()
print(cd)
# print(ed)