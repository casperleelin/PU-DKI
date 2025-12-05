#!/usr/bin/env python3
"""
Ultra-clean roughness comparison – ONE SCALAR per cloud
Fixed radius → no density bias
Perfect for papers and reports
"""

import open3d as o3d
import numpy as np
import argparse
import pandas as pd

def global_roughness_fixed_radius(pcd, radius=0.005, min_pts=10):
    points = np.asarray(pcd.points)
    tree = o3d.geometry.KDTreeFlann(pcd)
    
    all_dists = []
    
    for i in range(len(points)):
        # Fixed radius neighborhood
        [_, idx, _] = tree.search_radius_vector_3d(points[i], radius)
        idx = np.asarray(idx)
        if i in idx:
            idx = idx[idx != i]
        if len(idx) < min_pts:
            continue  # skip isolated points
            
        local_pts = points[idx] - points[i]
        
        # Best-fit plane
        try:
            _, _, vh = np.linalg.svd(local_pts, full_matrices=False)
            normal = vh[2, :]
            normal /= np.linalg.norm(normal)
            distances = np.abs(local_pts @ normal)
            all_dists.extend(distances)
        except:
            continue
    
    if len(all_dists) == 0:
        return {k: np.nan for k in ['Ra','Rq','Rz','points_used']}
    
    arr = np.array(all_dists)
    return {
        'Ra': arr.mean(),
        'Rq': np.sqrt(np.mean(arr**2)),
        'Rz': arr.max() - arr.min(),
        'points_used': len(arr)
    }

def main():
    parser = argparse.ArgumentParser(description="One scalar roughness per cloud – bias-free")
    parser.add_argument("--original", required=True, help="Original sparse cloud")
    parser.add_argument("--upsampled", required=True, help="Upsampled dense cloud")
    parser.add_argument("--radius", type=float, default=0.005, help="Radius in meters (e.g. 0.005 = 5 mm)")
    parser.add_argument("--multi", action="store_true", help="Test 2,5,10,20 mm")
    args = parser.parse_args()

    pcd_o = o3d.io.read_point_cloud(args.original)
    pcd_u = o3d.io.read_point_cloud(args.upsampled)
    
    radii = [args.radius] if not args.multi else [0.002, 0.005, 0.010, 0.020]
    
    print(f"\n{'Radius':>6}  {'Cloud':>10}  {'Rq (mm)':>10}  {'Ra (mm)':>10}  {'Rz (mm)':>10}  {'Pts used':>12}")
    print("-" * 68)
    
    results = []
    for r in radii:
        r_mm = r * 1000
        stat_o = global_roughness_fixed_radius(pcd_o, r)
        stat_u = global_roughness_fixed_radius(pcd_u, r)
        
        delta = (stat_u['Rq'] - stat_o['Rq']) / stat_o['Rq'] * 100 if stat_o['Rq'] > 0 else 0
        
        print(f"{r_mm:6.0f}mm  {'Original':>10}  {stat_o['Rq']*1000:10.4f}  {stat_o['Ra']*1000:10.4f}  {stat_o['Rz']*1000:10.4f}  {stat_o['points_used']:12,d}")
        print(f"{'':>16}  {'Upsampled':>10}  {stat_u['Rq']*1000:10.4f}  {stat_u['Ra']*1000:10.4f}  {stat_u['Rz']*1000:10.4f}  {stat_u['points_used']:12,d}")
        print(f"{'':>38}  ΔRq = {delta:+6.1f}%")
        print()
        
        results.append({
            'radius_mm': r_mm,
            'cloud': 'original', 'Rq_mm': stat_o['Rq']*1000, 'Ra_mm': stat_o['Ra']*1000,
            'Rz_mm': stat_o['Rz']*1000, 'points_used': stat_o['points_used']
        })
        results.append({
            'radius_mm': r_mm,
            'cloud': 'upsampled', 'Rq_mm': stat_u['Rq']*1000, 'Ra_mm': stat_u['Ra']*1000,
            'Rz_mm': stat_u['Rz']*1000, 'points_used': stat_u['points_used']
        })
    
    pd.DataFrame(results).to_csv("roughness_scalar_summary_outcrop1.csv", index=False)
    print("Scalar results saved → roughness_scalar_summary.csv")

if __name__ == "__main__":
    main()