import os
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import precision_score, recall_score
from sklearn.cluster import DBSCAN
from argparse import ArgumentParser
from ligmet.utils.constants import metals
def write_pdb_with_grids(
    pdb_id,
    metal_positions,
    grid_positions,
    grid_predictions,
    grid_type_predictions,
    pdb_input_dir,
    pdb_output_dir,
    pred_threshold=0.5,
):
    """
    1) 원본 PDB 파일(pdb_id.pdb)을 읽어서
    2) metal pred >= pred_threshold 를 만족하는 grid 좌표에 대해
       HETATM 라인을 추가해 저장.
    """
    os.makedirs(pdb_output_dir, exist_ok=True)

    input_pdb_path = os.path.join(pdb_input_dir, f"{pdb_id}.pdb")
    output_pdb_path = os.path.join(pdb_output_dir, f"{pdb_id}.pdb")

    if not os.path.exists(input_pdb_path):
        print(f"[WARNING] {input_pdb_path} not found. Skipping.")
        return

    with open(input_pdb_path, "r") as infile:
        pdb_lines = []
        for line in infile:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                pdb_lines.append(line)

    with open(output_pdb_path, "w") as outfile:
        # 1) 기존 PDB 내용 먼저 기록
        for line in pdb_lines:
            outfile.write(line)

        # 2) 조건을 만족하는 그리드 좌표 기록
        start_idx = 0  # 임의로 레지듀 번호 시작
        for idx, (grid_pos, grid_pred, grid_type_pred) in enumerate(
            zip(grid_positions, grid_predictions, grid_type_predictions)
        ):
            if grid_pred >= pred_threshold:
                metal_type_idx = torch.argmax(torch.tensor(grid_type_pred[:-1])).item()
                if metal_type_idx < len(metals):
                    metal_type = metals[metal_type_idx]
                else:
                    metal_type = "UNK"

                atom_idx = start_idx + idx
                x, y, z = grid_pos
                outfile.write(
                    f"HETATM{atom_idx:>5} {metal_type:<3}  GRD A{atom_idx:>4}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  {grid_pred:.2f}  0.00         {metal_type:>3}\n"
                )

    print(f"[INFO] Saved PDB with grids: {output_pdb_path}")




def main():
    pdb_id = '8xxa'
    result_path = f"/home/qkrgangeun/LigMet/data/biolip/test/0526/test_last_{pdb_id}.npz"
    pdb_input_dir = "/home/qkrgangeun/LigMet/data/biolip/merged"
    pdb_output_dir = "/home/qkrgangeun/LigMet/data/biolip/test/0526/pdbs"
    # grid_path = f"/home/qkrgangeun/LigMet/data/biolip/dl/features/{pdb_id}.npz"
    os.makedirs(pdb_output_dir, exist_ok=True)
    
    data = np.load(result_path, allow_pickle=True)
    # feat = np.load(grid_path, allow_pickle=True)
    
    metal_positions = data['metal_positions']
    metal_types = data['metal_types']
    grid_positions = data['grid_positions']
    grid_predictions = data['pred']
    grid_type_predictions = data['type_pred']
    
    write_pdb_with_grids(
    pdb_id,
    metal_positions,
    grid_positions,
    grid_predictions,
    grid_type_predictions,
    pdb_input_dir,
    pdb_output_dir,
    pred_threshold=0.1,  # 필요에 따라 수정
)


if __name__ == "__main__":
    main()