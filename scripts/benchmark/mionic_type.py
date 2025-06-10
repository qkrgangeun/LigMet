import os
import csv
import ast
import numpy as np
from collections import defaultdict, Counter
from scipy.spatial.distance import cdist
from ligmet.utils.constants import metals

# ------------------------------
# 설정
# ------------------------------
binding_csv_path = "/home/qkrgangeun/LigMet/benchmark/mionic/metal_binding_sites_residx_NOSSE_3.0.csv"
prediction_dir = "/home/qkrgangeun/LigMet/benchmark/mionic/testset_result"
pdb_list_file = "/home/qkrgangeun/LigMet/data/biolip_backup/pdb/test_pdb_noerror.txt"
npz_dir = "/home/qkrgangeun/LigMet/data/biolip_backup/test/0602"
excluded_metals = {"PO4", "SO4"}
threshold = 0.001

# ------------------------------
# 유효한 PDB 리스트 불러오기
# ------------------------------
with open(pdb_list_file, "r") as f:
    valid_pdb_ids = set(line.strip().lower() for line in f if line.strip())

# ------------------------------
# 바인딩 메탈 위치 불러오기
# ------------------------------
binding_metal_positions = defaultdict(set)
with open(binding_csv_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        pdb_id = row["PDB ID"].strip().lower()
        if pdb_id not in valid_pdb_ids:
            continue
        try:
            pos = ast.literal_eval(row["Metal Position"])  # [x, y, z]
            pos_tuple = tuple(round(float(x), 3) for x in pos)
        except:
            continue
        binding_metal_positions[pdb_id].add(pos_tuple)

# ------------------------------
# Residue-level 예측 로딩
# ------------------------------
residue_predictions = defaultdict(dict)
for csv_file in os.listdir(prediction_dir):
    if not csv_file.endswith("_result.csv"):
        continue
    parts = csv_file.split("_")
    if len(parts) < 3:
        continue
    pdb_id = parts[1][:-1].lower()
    chain_id = parts[1][-1]
    if pdb_id not in valid_pdb_ids:
        continue
    csv_path = os.path.join(prediction_dir, csv_file)
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            idx = i + 1  # 1-based indexing
            scores = {k: float(v) for k, v in row.items() if k not in {"lab", "PO4", "SO4"}}
            if not scores or max(scores.values()) < threshold:
                continue
            top1 = max(scores.items(), key=lambda x: x[1])[0]
            residue_predictions[(pdb_id, chain_id)][idx] = top1

# ------------------------------
# 예측 결과 집계
# ------------------------------
metal_true_types = defaultdict(list)
metal_predicted_types = defaultdict(list)

with open(binding_csv_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        pdb_id = row["PDB ID"].strip().lower()
        metal_type = row["Metal Type"].strip().upper()
        if pdb_id not in valid_pdb_ids or metal_type in excluded_metals:
            continue

        try:
            metal_pos = tuple(round(float(x), 3) for x in ast.literal_eval(row["Metal Position"]))
            residues = ast.literal_eval(row["Binding Residues"])
            chains = ast.literal_eval(row["Binding Chains"])
        except Exception as e:
            print(f"Error parsing row in {pdb_id}: {e}")
            continue


        # .npz 파일에서 메탈 위치 일치하는지 확인
        result_path = os.path.join(npz_dir, f"test_{pdb_id}.npz")
        if not os.path.exists(result_path):
            continue

        data = np.load(result_path, allow_pickle=True)
        npz_metal_positions = data["metal_positions"]  # (N, 3)
        npz_metal_types = data["metal_types"]          # (N,)

        # 위치 매칭
        match_found = False
        for i, pos in enumerate(npz_metal_positions):
            pos_tuple = tuple(round(float(x), 3) for x in pos)
            if pos_tuple == metal_pos:
                match_found = True
                break

        if not match_found:
            continue  # 이 metal은 npz와 위치 불일치

        # residue 예측에서 majority type 수집
        pred_types = []
        for res in residues:
            if ':' not in res:
                continue
            _, residx = res.split(':')
            try:
                residx = int(residx)
            except:
                continue
            for chain in chains:
                pred_type = residue_predictions.get((pdb_id, chain), {}).get(residx)
                if pred_type:
                    pred_types.append(pred_type)
                    break

        majority_type = (
            Counter(pred_types).most_common(1)[0][0] if pred_types else "__GARBAGE__"
        )

        metal_true_types[metal_type].append(metal_type)
        metal_predicted_types[metal_type].append(majority_type)

# ------------------------------
# 평가 결과 출력
# ------------------------------
print(f"{'Metal':<6} | {'Recall':>6} | TP / Total")
print("-" * 25)
for metal in sorted(metal_true_types):
    total = len(metal_true_types[metal])
    correct = sum(1 for p in metal_predicted_types[metal] if p == metal)
    recall = correct / total if total > 0 else 0.0
    print(f"{metal:<6} | {recall:>6.3f} | {correct} / {total}")
