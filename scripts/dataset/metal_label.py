import os
import numpy as np
import pandas as pd
import ast
from joblib import Parallel, delayed

# 파일 경로 설정
pdb_dir = '/home/qkrgangeun/LigMet/data/biolip/pdb'
mbs_path = "/home/qkrgangeun/LigMet/code/text/biolip/all_metal_binding_sites_NOSSE_3.0.csv"
data_txt = '/home/qkrgangeun/LigMet/benchmark/test_pdb.txt'
save_dir = "/home/qkrgangeun/LigMet/data/biolip/metal_label"

# 저장 디렉토리가 없으면 생성
os.makedirs(save_dir, exist_ok=True)

# 금속 결합부 정보 CSV 파일 읽기
df = pd.read_csv(mbs_path)
unique_pdb_ids = df['PDB ID'].unique()
# train_val_test.txt 파일에서 PDB ID 읽기 (한 줄에 하나씩 있다고 가정)
with open(data_txt, 'r') as f:
    pdb_ids = [line.strip() for line in f if line.strip()]

def process_pdb(pdb_id, df, save_dir):
    """
    주어진 PDB ID에 대해 CSV 파일에서 금속 결합부 정보를 추출하고,
    binding residues 개수가 3개 이상인 경우에만, metal_positions (n, 3) 및 metal_types (n,) 배열로 변환하여 npz로 저장합니다.
    """
    # 해당 PDB ID에 해당하는 행 필터링
    df_pdb = df[df["PDB ID"] == pdb_id]
    
    metal_positions_list = []
    metal_types_list = []
    
    for _, row in df_pdb.iterrows():
        # Binding Residues 컬럼에 대해 문자열을 리스트로 변환 후 원소 개수 확인
        binding_residues_str = row["Binding Residues"]
        try:
            binding_residues = ast.literal_eval(binding_residues_str)
            if not isinstance(binding_residues, list) or len(binding_residues) < 3:
                # 잔기 개수가 3개 미만이면 이 행은 건너뜁니다.
                continue
        except Exception as e:
            print(f"PDB {pdb_id}: binding residues 파싱 오류 -> {binding_residues_str}")
            continue
        
        # Metal Position 컬럼 처리 (문자열을 리스트로 변환)
        metal_position_str = row["Metal Position"]
        try:
            metal_position = ast.literal_eval(metal_position_str)
            if not isinstance(metal_position, list) or len(metal_position) != 3:
                print(f"PDB {pdb_id}: 잘못된 metal position 형식 -> {metal_position_str}")
                continue
        except Exception as e:
            print(f"PDB {pdb_id}: metal position 파싱 오류 -> {metal_position_str}")
            continue
        
        metal_positions_list.append(metal_position)
        metal_types_list.append(row["Metal Type"])
    
    # 해당 PDB ID에 유효한 정보가 없으면 저장하지 않습니다.
    if len(metal_positions_list) == 0:
        print(f"PDB {pdb_id}: 조건에 부합하는 금속 결합부 정보가 없습니다.")
        return
    
    # 리스트를 numpy 배열로 변환
    metal_positions = np.array(metal_positions_list)  # (n, 3)
    metal_types = np.array(metal_types_list)          # (n,)
    
    # npz 파일로 저장 (파일 이름은 pdb_id로 저장)
    save_path = os.path.join(save_dir, f"{pdb_id}.npz")
    np.savez(save_path, metal_positions=metal_positions, metal_types=metal_types)
    print(f"PDB {pdb_id}: {metal_types} {metal_positions.shape[0]}개의 결합부 정보를 {save_path}에 저장하였습니다.")

# joblib의 Parallel을 사용하여 병렬 처리 (n_jobs=-1: 사용 가능한 모든 코어 사용)
Parallel(n_jobs=-1)(delayed(process_pdb)(pdb_id, df, save_dir) for pdb_id in unique_pdb_ids)
