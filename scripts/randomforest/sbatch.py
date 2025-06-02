#!/usr/bin/env python3
import os

# pdb_id 목록 파일 경로들
pdb_files = [
    # "/home/qkrgangeun/LigMet/code/text/biolip/train_pdbs.txt",
    # "/home/qkrgangeun/LigMet/code/text/biolip/val_pdbs.txt",
    # "/home/qkrgangeun/LigMet/code/text/biolip/test_pdbs.txt"
    # "/home/qkrgangeun/LigMet/data/biolip/rf/no.txt"
    "/home/qkrgangeun/LigMet/code/notebooks/error_pdbs.txt"
]

# pdb_id를 모아서 중복 제거
pdb_ids = set()
for file_path in pdb_files:
    with open(file_path, "r") as f:
        for line in f:
            pdb_id = line.strip()
            if pdb_id:
                pdb_ids.add(pdb_id)

# 출력 스크립트 저장 디렉토리
output_dir = "/home/qkrgangeun/LigMet/sh/rl"
os.makedirs(output_dir, exist_ok=True)

# 각 pdb_id별로 SLURM 스크립트 생성
for pdb_id in sorted(pdb_ids):
    script_content = f"""#!/bin/sh
#SBATCH -J {pdb_id}
#SBATCH -p local.q
#SBATCH -w galaxy4
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -o /home/qkrgangeun/LigMet/sh/rl/error_{pdb_id}.log
#SBATCH -e /home/qkrgangeun/LigMet/sh/rl/error_{pdb_id}.log
python /home/qkrgangeun/LigMet/code/scripts/randomforest/make_rf_features.py /home/qkrgangeun/LigMet/data/biolip/dl/features/{pdb_id}.npz --output_dir /home/qkrgangeun/LigMet/data/biolip/rl/features
python /home/qkrgangeun/LigMet/code/scripts/randomforest/test_rf.py --model_path "/home/qkrgangeun/LigMet/data/rf_param/0415_newlabel_chain1" --test_data {pdb_id}
"""
    output_file = os.path.join(output_dir, f"error_{pdb_id}.sh")
    with open(output_file, "w") as f:
        f.write(script_content)
    print(f"Created script for {pdb_id} at {output_file}")

# python /home/qkrgangeun/LigMet/code/scripts/randomforest/make_rf_features.py /home/qkrgangeun/LigMet/data/biolip/dl/features/{pdb_id}.npz --output_dir /home/qkrgangeun/LigMet/data/biolip/rl/features