import os
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
import argparse

def compute_precision_recall(predicted_coords, true_coords, dist_threshold=2.0):
    if len(predicted_coords) == 0:
        return 0.0, 0.0 if len(true_coords) > 0 else (1.0, 1.0)
    
    tp = 0
    fp = 0
    matched_true = []

    for pred_coord in predicted_coords:
        distances = np.linalg.norm(true_coords - pred_coord, axis=1)
        if np.any(distances < dist_threshold):
            tp += 1
            matched_true.append(np.argmin(distances))
        else:
            fp += 1

    fn = len(true_coords) - len(set(matched_true))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return precision, recall

def dbscan_clustering(coords, preds, eps=2.0, min_samples=1, method='max'):
    if len(coords) == 0:
        return np.array([]), np.array([])

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = db.labels_
    unique_labels = set(labels)
    representatives = []

    for label in unique_labels:
        if label == -1:
            continue
        indices = np.where(labels == label)[0]
        if method == 'max':
            best_idx = indices[np.argmax(preds[indices])]
            representatives.append(coords[best_idx])
        elif method == 'mean':
            cluster_mean = np.mean(coords[indices], axis=0)
            representatives.append(cluster_mean)
        else:
            raise ValueError("method must be 'max' or 'mean'")

    return np.array(representatives), labels

def main(pdb_id, score_threshold):
    result_path = f"/home/qkrgangeun/LigMet/data/biolip/test/0526/test_last_{pdb_id}.npz"
    if not os.path.exists(result_path):
        print(f"[!] File not found: {result_path}")
        return None, None

    data = np.load(result_path, allow_pickle=True)

    metal_positions = data['metal_positions']
    grid_positions = data['grid_positions']
    grid_predictions = data['pred']

    mask = grid_predictions >= score_threshold
    selected_positions = grid_positions[mask]
    selected_preds = grid_predictions[mask]

    rep_positions, _ = dbscan_clustering(selected_positions, selected_preds, eps=2.0, min_samples=2, method='max')
    precision, recall = compute_precision_recall(rep_positions, metal_positions, dist_threshold=2.0)

    print(f"[{pdb_id}] Precision: {precision:.3f}, Recall: {recall:.3f}")
    return precision, recall

def run_batch(pdb_list, score_threshold):
    precisions = []
    recalls = []

    for pdb_id in pdb_list:
        precision, recall = main(pdb_id.strip(), score_threshold)
        if precision is not None and recall is not None:
            precisions.append(precision)
            recalls.append(recall)

    if precisions:
        mean_precision = np.mean(precisions)
        mean_recall = np.mean(recalls)
        print(f"\n==> Mean Precision: {mean_precision:.3f}, Mean Recall: {mean_recall:.3f}")

    else:
        print("[!] No valid results.")
    return precisions, recalls

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch evaluation using DBSCAN clustering and precision/recall metrics")
    parser.add_argument("--pdb_list_file", type=str, required=True, help="Path to text file containing PDB IDs (one per line)")
    parser.add_argument("--score_threshold", type=float, default=0.5, help="Prediction score threshold (default=0.5)")

    args = parser.parse_args()

    if not os.path.isfile(args.pdb_list_file):
        raise FileNotFoundError(f"PDB list file not found: {args.pdb_list_file}")

    with open(args.pdb_list_file, 'r') as f:
        pdb_list = [line.strip() for line in f if line.strip()]

    run_batch(pdb_list, args.score_threshold)
