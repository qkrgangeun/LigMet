import os
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cdist
from ligmet.utils.constants import metals  # e.g., ['ZN', 'CA', 'MG', ...]

def evaluate_single_pdb(pdb_id, score_threshold=0.5, top_k=1):
    result_path = f"/home/qkrgangeun/LigMet/data/biolip_backup/test/0602/test_{pdb_id}.npz"
    if not os.path.exists(result_path):
        print(f"[!] File not found: {result_path}")
        return [], []
    else:
        print(f"[✔️] Evaluating {pdb_id}...")
        data = np.load(result_path, allow_pickle=True)

        metal_positions = data['metal_positions']         # (N_metal, 3)
        metal_types = data['metal_types']                 # (N_metal,) → e.g., [0, 1, 2, ...] (int indices of metals)
        grid_positions = data['grid_positions']           # (N_grid, 3)
        grid_predictions = data['pred']                   # (N_grid,)
        grid_type_probs = data['type_pred']               # (N_grid, 10)

        # Threshold filtering
        num_grid = len(grid_positions) - len(metal_positions)
        mask = grid_predictions >= score_threshold
        mask2 = np.zeros_like(mask, dtype=bool)
        mask2[:num_grid] = True  # 앞쪽 grid만 True
        final_mask = mask & mask2
        filtered_positions = grid_positions[final_mask]
        filtered_type_probs = grid_type_probs[final_mask]

        if len(filtered_positions) == 0:
            return [], []

        # Compute distances between each metal and all grid points
        dists = cdist(metal_positions, filtered_positions)
        closest_idx = np.argmin(dists, axis=1)

        predictions = []
        labels = []
        for i, true_type in enumerate(metal_types):
            pred_probs = filtered_type_probs[closest_idx[i]]
            topk_pred_indices = np.argsort(pred_probs)[::-1][:top_k]

            print('true_type',true_type)
            true_idx = true_type  # int → int index


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

    # Compute per-class accuracy (recall)
    per_class_accuracy = {
        metals[k]: correct_by_type[k] / total_by_type[k] if total_by_type[k] > 0 else 0.0
        for k in total_by_type
    }

    return overall_accuracy, per_class_accuracy, total_by_type


def run_batch(pdb_list, score_threshold=0.5, top_k=1):
    all_predictions = []
    all_labels = []

    for pdb_id in pdb_list:
        preds, lbls = evaluate_single_pdb(pdb_id, score_threshold, top_k)
        print(preds,lbls)
        all_predictions.extend(preds)
        all_labels.extend(lbls)

    overall_acc, per_class_acc, total_by_type = compute_topk_accuracy(all_predictions, all_labels, top_k)

    print(f"\n[🔎 Evaluation Results | Top-{top_k} | Score threshold = {score_threshold}]")
    print(f"Overall Accuracy (≒ Recall): {overall_acc:.4f}")

    print("\n[Per-metal Recall and Count]")
    print(per_class_acc)
    for metal_name in sorted(per_class_acc.keys()):
        recall = per_class_acc[metal_name]
        count = total_by_type[metals.index(metal_name)]
        print(f"{metal_name}: Recall = {recall:.4f}, Count = {count}")


if __name__ == "__main__":
    pdb_list_file = "/home/qkrgangeun/LigMet/data/biolip_backup/pdb/test_pdb_noerror.txt"
    
    # PDB 리스트 불러오기
    with open(pdb_list_file, 'r') as f:
        pdb_list = [line.strip() for line in f if line.strip()]
    
    # 예시: 특정 샘플만 평가하고 싶을 경우
    # pdb_list = ['8yk5']

    # 평가 실행
    run_batch(pdb_list, score_threshold=0.0, top_k=1)
