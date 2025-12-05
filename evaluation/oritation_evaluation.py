# normal_comparison.py
import open3d as o3d
import numpy as np
import argparse
import os

def load_pcd(path):
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        raise ValueError(f"Empty point cloud: {path}")
    if not pcd.has_normals():
        raise ValueError(f"No normals in {path}. Run normal estimation first!")
    return pcd

def compute_normal_metrics(pcd_ref, pcd_pred, knn=10, max_distance=0.05):
    """
    Compare normals between reference (ground truth) and predicted point cloud.
    Uses nearest-neighbor correspondence (standard practice).
    """
    points_ref = np.asarray(pcd_ref.points)
    points_pred = np.asarray(pcd_pred.points)
    normals_ref = np.asarray(pcd_ref.normals)
    normals_pred = np.asarray(pcd_pred.normals)

    # Normalize just in case
    normals_ref = normals_ref / (np.linalg.norm(normals_ref, axis=1, keepdims=True) + 1e-8)
    normals_pred = normals_pred / (np.linalg.norm(normals_pred, axis=1, keepdims=True) + 1e-8)

    # Build KDTree on reference
    pcd_ref_tree = o3d.geometry.KDTreeFlann(pcd_ref)

    angles_deg = []
    cosines = []

    print(f"Comparing {len(points_pred)} predicted points → nearest ref point (max dist = {max_distance})")

    valid_count = 0
    for i, pt in enumerate(points_pred):
        [_, idx, dist2] = pcd_ref_tree.search_knn_vector_3d(pt, knn)
        dist = np.sqrt(dist2[0])

        if dist > max_distance:
            continue  # ignore points too far (common in sparse/dense comparison)

        n_ref = normals_ref[idx[0]]
        n_pred = normals_pred[i]

        cosine = np.dot(n_ref, n_pred)
        cosine = np.clip(cosine, -1.0, 1.0)
        angle_rad = np.arccos(cosine)
        angle_deg = np.degrees(angle_rad)

        cosines.append(cosine)
        angles_deg.append(angle_deg)
        valid_count += 1

    if valid_count == 0:
        raise ValueError("No corresponding points found! Check scale/alignment.")

    angles = np.array(angles_deg)
    cosines = np.array(cosines)

    # Compute metrics
    mean_angle = angles.mean()
    median_angle = np.median(angles)
    nc = cosines.mean()                                   # Normal Consistency
    percent_below_30 = 100.0 * (angles < 30.0).sum() / len(angles)
    percent_below_15 = 100.0 * (angles < 15.0).sum() / len(angles)

    return {
        "Normal Consistency": nc,
        "Mean Angle (°)": mean_angle,
        "Median Angle (°)": median_angle,
        "% < 15°": percent_below_15,
        "% < 30°": percent_below_30,
        "Valid correspondences": valid_count
    }

# ————————————————————————
# Main
# ————————————————————————
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare surface orientation between two point clouds")
    parser.add_argument("-r", "--ref", type=str, required=True, help="Reference (GT) point cloud with normals")
    parser.add_argument("-p", "--pred", type=str, required=True, help="Predicted point cloud with normals")
    parser.add_argument("--max_dist", type=float, default=0.05,
                        help="Max distance for correspondence (adjust to point cloud scale, PU1K ≈ 0.03–0.06)")
    parser.add_argument("--knn", type=int, default=10, help="K for nearest neighbor search")

    args = parser.parse_args()

    print("Loading point clouds...")
    pcd_ref = load_pcd(args.ref)
    pcd_pred = load_pcd(args.pred)

    print(f"Reference : {len(pcd_ref.points)} points")
    print(f"Predicted : {len(pcd_pred.points)} points")

    metrics = compute_normal_metrics(
        pcd_ref, pcd_pred,
        knn=args.knn,
        max_distance=args.max_dist
    )

    print("\n" + "="*50)
    print("          SURFACE ORIENTATION COMPARISON")
    print("="*50)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:25}: {v:.6f}")
        else:
            print(f"{k:25}: {v}")
    print("="*50)

    # Optional: visualize side-by-side with normal colors
    vis = False
    if vis:
        pcd_ref.paint_uniform_color([0.8, 0.0, 0.0])  # red = GT
        pcd_pred.paint_uniform_color([0.0, 0.8, 0.0])  # green = pred
        o3d.visualization.draw_geometries_with_editing([pcd_ref, pcd_pred],
                                                        window_name="Red=GT, Green=Pred")