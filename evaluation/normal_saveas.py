import numpy as np
import open3d as o3d
import os

def meshlab_xyz_to_o3d_with_normals(input_path, output_path=None):
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    # This line fixes 95% of weird files
    data = np.loadtxt(input_path, dtype=np.float32, comments=None, delimiter=None)

    if data.shape[1] not in (3, 6):
        raise ValueError(f"Expected 3 or 6 columns, got {data.shape[1]}. Check your .xyz file!")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[:, :3])

    if data.shape[1] == 6:
        normals = data[:, 3:6]
        # Fix non-unit normals (MeshLab sometimes outputs garbage lengths)
        norm_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / (norm_lengths + 1e-8)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        print(f"Loaded {len(pcd.points)} points with normals (max length was {norm_lengths.max():.4f})")
    else:
        print("Only XYZ → computing normals with Open3D")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.orient_normals_to_align_with_direction([0,0,1])

    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + "_with_o3d.ply"
    
    o3d.io.write_point_cloud(output_path, pcd, write_ascii=False)
    print(f"Saved → {output_path}")
    return pcd

# Usage
pcd = meshlab_xyz_to_o3d_with_normals("real_rock_outcrop/Real_PUCRN/limestone_3056_normal.xyz")