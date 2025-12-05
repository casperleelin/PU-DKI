# compare_two_segmentations_no_gt.py
import numpy as np
import open3d as o3d
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix
import argparse

def load_labeled_point_cloud(path):
    """Load point cloud that already has integer labels (no separate label file needed)"""
    if path.endswith('.ply'):
        pcd = o3d.io.read_point_cloud(path)
        points = np.asarray(pcd.points)
        if pcd.has_colors():
            # Many people store labels as RGB → take red channel or grayscale
            colors = np.asarray(pcd.colors)
            labels = np.round(colors[:, 0] * 255).astype(int)  # assuming labels encoded in red
        else:
            raise ValueError("PLY has no color/label field")
        return points, labels
    else:
        # .txt, .xyz, .las etc. with label in last column
        data = np.loadtxt(path)
        return data[:, :3], data[:, -1].astype(int)

def rand_index(labels1, labels2):
    tn = tp = fn = fp = 0
    n = len(labels1)
    for i in range(n):
        for j in range(i+1, n):
            same1 = labels1[i] == labels1[j]
            same2 = labels2[i] == labels2[j]
            if same1 and same2:  tp += 1
            if same1 and not same2: fp += 1
            if not same1 and same2: fn += 1
            if not same1 and not same2: tn += 1
    ri = (tp + tn) / (tp + tn + fp + fn)
    return ri

def variation_of_information(labels1, labels2):
    # Fast implementation using contingency matrix
    cm = contingency_matrix(labels1, labels2)
    a = np.sum(cm, axis=1)
    b = np.sum(cm, axis=0)
    n = np.sum(cm)
    joint = cm / n
    marg1 = a / n
    marg2 = b / n
    h1 = -np.sum(marg1 * np.log(marg1 + 1e-12))
    h2 = -np.sum(marg2 * np.log(marg2 + 1e-12))
    mi = np.sum(joint * np.log(joint / (np.outer(marg1, marg2) + 1e-12) + 1e-12))
    voi = h1 + h2 - 2 * mi
    return voi

def segmentation_covering(labels_ref, labels_query):
    """How well query segmentation is covered by ref (higher = better)"""
    cm = contingency_matrix(labels_ref, labels_query)
    row_sums = cm.sum(axis=1)
    covering = np.sum(cm.max(axis=1)) / row_sums.sum()
    return covering

def main():
    parser = argparse.ArgumentParser(description="Compare TWO unlabeled segmented rock point clouds")
    parser.add_argument('--pcd1', required=True, help='First segmented point cloud (with labels in color or last column)')
    parser.add_argument('--pcd2', required=True, help='Second segmented point cloud')
    parser.add_argument('--min_points', type=int, default=30, help='Ignore tiny clusters')
    args = parser.parse_args()

    print("Loading first segmentation...")
    pts1, lab1 = load_labeled_point_cloud(args.pcd1)
    print("Loading second segmentation...")
    pts2, lab2 = load_labeled_point_cloud(args.pcd2)

    # Make sure we compare the exact same points
    if len(pts1) != len(pts2) or not np.allclose(pts1, pts2):
        print("Point clouds differ in size or coordinates → aligning by nearest neighbor...")
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pts1)
        tree = o3d.geometry.KDTreeFlann(pcd1)
        lab2_aligned = np.zeros(len(pts1), dtype=int) - 1
        for i, pt in enumerate(pts2):
            [_, idx, _] = tree.search_knn_vector_3d(pt, 1)
            lab2_aligned[idx[0]] = lab2[i]
        lab2 = lab2_aligned

    # Remove noise label (-1 or 0) and tiny clusters
    mask = (lab1 > 0) & (lab2 > 0)
    lab1 = lab1[mask]
    lab2 = lab2[mask]

    # Optional: filter tiny clusters
    unique1, counts1 = np.unique(lab1, return_counts=True)
    unique2, counts2 = np.unique(lab2, return_counts=True)
    valid1 = set(unique1[counts1 >= args.min_points])
    valid2 = set(unique2[counts2 >= args.min_points])
    mask = np.isin(lab1, list(valid1)) & np.isin(lab2, list(valid2))
    lab1 = lab1[mask]
    lab2 = lab2[mask]

    # Compute all metrics
    ari = adjusted_rand_score(lab1, lab2)
    ri  = rand_index(lab1, lab2)                       # regular Rand Index
    voi = variation_of_information(lab1, lab2)
    sc1 = segmentation_covering(lab1, lab2)  # how well seg2 covers seg1
    sc2 = segmentation_covering(lab2, lab1)  # and vice versa

    print("\n" + "="*60)
    print("UNSUPERVISED SEGMENTATION COMPARISON (no ground truth needed)")
    print("="*60)
    print(f"Adjusted Rand Index (ARI)      : {ari:.4f}   (higher is better, max=1.0)")
    print(f"Rand Index                     : {ri:.4f}   (higher is better)")
    print(f"Variation of Information (VoI) : {voi:.4f}   (lower is better, min=0)")
    print(f"SegCovering A→B                : {sc1:.4f}   (how well B covers A)")
    print(f"SegCovering B→A                : {sc2:.4f}   (how well A covers B)")
    print(f"Mean Covering                  : {(sc1+sc2)/2:.4f}")
    print(f"Points used for comparison     : {len(lab1)}")
    print("="*60)

if __name__ == "__main__":
    main()