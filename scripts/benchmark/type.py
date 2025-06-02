import os
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cdist
from ligmet.utils.constants import metals

def evaluate_single_pdb(pdb_id, score_threshold=0.5, top_k=1):
    result_path = f"/home/qkrgangeun/LigMet/data/biolip/test/0526/test_last_{pdb_id}.npz"
    if not os.path.exists(result_path):
        print(f"[!] File not found: {result_path}")
        return [], []

    data = np.load(result_path, allow_pickle=True)

    metal_positions = data['metal_positions']          # (N_metal, 3)
    metal_types = data['metal_types']                  # (N_metal,)
    grid_positions = data['grid_positions']            # (N_grid, 3)
    grid_predictions = data['pred']                    # (N_grid,)
    grid_type_probs = data['pred_types']               # (N_grid, 10) softmax over metal types

    # Threshold filtering
    mask = grid_predictions >= score_threshold
    filtered_positions = grid_positions[mask]
    filtered_type_probs = grid_type_probs[mask]

    if len(filtered_positions) == 0:
        return [], []

    # Compute distances between each metal and all grid points
    dists = cdist(metal_positions, filtered_positions)  # shape (N_metal, N_selected_grid)
    closest_idx = np.argmin(dists, axis=1)

    # Get predicted type for the closest grid
    predictions = []
    labels = []
    for i, true_type in enumerate(metal_types):
        pred_probs = filtered_type_probs[closest_idx[i]]
        topk_pred_indices = np.argsort(pred_probs)[::-1][:top_k]
        true_idx = metals.index(true_type)

        predictions.append((true_idx, topk_pred_indices))
        labels.append(true_idx)

    return predictions, labels


def compute_topk_accuracy(predictions, labels, top_k=1):
    correct_by_type = defaultdict(int)
    total_by_type = defaultdict(int)
    total_correct = 0

    for (true_idx, topk_pred), label in zip(predictions, labels):
        total_by_type[label] += 1
        if label in topk_pred:
            correct_by_type[label] += 1
            total_correct += 1

    # Compute overall accuracy
    total_samples = len(labels)
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    # Compute per-class accuracy
    per_class_accuracy = {
        metals[k]: correct_by_type[k] / total_by_type[k] if total_by_type[k] > 0 else 0.0
        for k in total_by_type
    }

    return overall_accuracy, per_class_accuracy


def run_batch(pdb_list, score_threshold=0.5, top_k=1):
    all_predictions = []
    all_labels = []

    for pdb_id in pdb_list:
        preds, lbls = evaluate_single_pdb(pdb_id, score_threshold, top_k)
        all_predictions.extend(preds)
        all_labels.extend(lbls)

    overall_acc, per_class_acc = compute_topk_accuracy(all_predictions, all_labels, top_k)

    print(f"\n[🔎 Evaluation Results | Top-{top_k} | Score threshold = {score_threshold}]")
    print(f"Overall Accuracy: {overall_acc:.4f}")
    print("\n[Per-metal Accuracy]")
    for metal, acc in per_class_acc.items():
        print(f"{metals[metal]}: {acc:.4f}")
