import open3d as o3d
import numpy as np
import os

def remove_utm_coordinates_open3d(point_cloud_file, output_file=None):
    """
    使用Open3D删除点云的UTM坐标信息
    将坐标转换为相对坐标或局部坐标系
    """
    # 读取点云
    pcd = o3d.io.read_point_cloud(point_cloud_file)
    points = np.asarray(pcd.points)
    
    print(f"原始点云点数: {len(points)}")
    print(f"原始坐标范围:")
    print(f"  X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"  Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"  Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    
    # 方法A: 转换为相对坐标（减去最小值）
    points_relative = points - points.min(axis=0)
    
    # 方法B: 转换为以质心为中心（推荐）
    centroid = points.mean(axis=0)
    points_centered = points - centroid
    
    # 方法C: 只保留高程信息（如果UTM主要在XY平面）
    # points_elevation = points.copy()
    # points_elevation[:, 0] = 0  # 清空X坐标（East）
    # points_elevation[:, 1] = 0  # 清空Y坐标（North）
    
    # 使用质心中心化方法
    pcd.points = o3d.utility.Vector3dVector(points_centered)
    
    print(f"\n转换后坐标范围:")
    print(f"  X: [{points_centered[:, 0].min():.2f}, {points_centered[:, 0].max():.2f}]")
    print(f"  Y: [{points_centered[:, 1].min():.2f}, {points_centered[:, 1].max():.2f}]")
    print(f"  Z: [{points_centered[:, 2].min():.2f}, {points_centered[:, 2].max():.2f}]")
    
    # 保存结果
    if output_file is None:
        base_name = os.path.splitext(point_cloud_file)[0]
        output_file = f"{base_name}_no_utm.ply"
    
    o3d.io.write_point_cloud(output_file, pcd)
    print(f"\n结果已保存: {output_file}")
    
    return pcd

# 使用示例
if __name__ == "__main__":
    input_file = "data/right.xyz"  # 替换为你的点云文件
    remove_utm_coordinates_open3d(input_file)