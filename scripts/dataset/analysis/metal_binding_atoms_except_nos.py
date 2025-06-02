import csv
import os
from pathlib import Path
from scipy.spatial import cKDTree
from multiprocessing import Pool
from ligmet.utils.pdb import read_pdb
import numpy as np

# Metal 원소 리스트 (참고용)
metals = {"ZN", "MG", "FE", "CA", "CU", "MN", "CO", "NI", "NA", "K"}

# PDB 파일이 저장된 디렉토리 (merged 폴더)
pdb_dir = Path("/home/qkrgangeun/LigMet/data/biolip/merged")
# 결과 CSV 파일 경로
output_csv = "/home/qkrgangeun/LigMet/code/text/biolip/metal_binding_sites_atom_except_nos.csv"
# Cluster 정보 CSV 파일 경로
cluster_csv = "/home/qkrgangeun/LigMet/code/text/biolip/clusterid_releaseddate.csv"


def read_cluster_info(cluster_csv):
    """
    cluster CSV 파일에서 pdb_id, chain_id, cluster 정보를 읽어와서
    (pdb_id, chain) -> cluster_id 형태의 사전(dict)으로 반환
    """
    cluster_info = {}
    with open(cluster_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdb_id = row['pdb_id']
            chains = eval(row['chains'])    # 문자열을 리스트로 변환
            clusters = eval(row['clusters'])  # 문자열을 리스트로 변환
            for chain, cluster in zip(chains, clusters):
                cluster_info[(pdb_id, chain)] = cluster
    return cluster_info


# cluster 정보를 전역 변수에 저장
cluster_info = read_cluster_info(cluster_csv)


def process_pdb(pdb_file, cutoff=3.0):
    """
    주어진 PDB 파일에서 각 금속 원자 주변 3 Å 이내의 원자들에 대해
    원자들의 원소 정보를 set으로 구성한 후, 그 set에 N, O, S, C 이외의 원소가
    하나라도 포함되면 해당 금속 정보와 주변 원자(리간드 관련) 정보를 반환합니다.

    반환하는 정보:
      - PDB ID
      - Metal Type
      - Metal Position (소수점 3자리 반올림)
      - Other Elements: 3 Å 이내의 원자들 중 N, O, S 제외한 원소들의 목록
      - Binding Chains: 해당 원소들을 포함하는 원자들이 속한 체인 (unique)
      - Cluster ID: 각 체인에 대응하는 cluster 정보
    """
    structure = read_pdb(pdb_file)

    # 금속 원자가 없는 경우 건너뛰기
    if structure.metal_positions is None or len(structure.metal_positions) == 0:
        print('no metal in', pdb_file)
        return []

    metal_positions = np.array(structure.metal_positions)      # (M, 3)
    metal_types = np.array(structure.metal_types)              # (M,)
    atom_positions = structure.atom_positions                  # (N, 3)
    atom_residues = structure.atom_residues                    # (N,) 잔기 이름 (예: 'GLU', 'ASP' 등)
    residue_idxs = structure.residue_idxs                      # (N,) 잔기 인덱스
    chain_ids = structure.chain_ids                            # (N,) 각 원자의 체인 정보
    atom_elements = np.array(structure.atom_elements)          # (N,) 각 원자의 원소 기호 (예: 'N', 'C', 'O', 'S' 등)

    # KDTree를 이용하여 각 금속 원자 주변 cutoff 거리 내의 원자 인덱스 찾기
    tree = cKDTree(atom_positions)
    neigh_indices = tree.query_ball_point(metal_positions, cutoff)  # 각 금속에 대해 이웃 원자 인덱스 목록

    results = []
    for metal_idx, metal_pos in enumerate(metal_positions):
        metal_type = metal_types[metal_idx]
        pdb_id = pdb_file.stem  # 파일명에서 확장자 제거
        neighbor_idx_list = neigh_indices[metal_idx]
        if not neighbor_idx_list:
            continue

        # 모든 이웃 원자의 원소 정보를 집합으로 구성 (대문자 처리)
        neighbor_elements_set = {atom_elements[i].upper() for i in neighbor_idx_list if atom_elements[i] is not None}
        # N, O, S 이외의 원소 찾기
        other_elements = neighbor_elements_set - {"N", "O", "S", "C"}
        if not other_elements:
            continue  # 다른 원소가 없으면 넘어감

        # 해당 금속 원자 주변에서 조건에 맞는 원소를 가진 원자들의 잔기 이름과 체인 정보 추출
        binding_residues = [atom_residues[i] for i in neighbor_idx_list if atom_elements[i].upper() in other_elements]
        binding_chains_raw = [chain_ids[i] for i in neighbor_idx_list if atom_elements[i].upper() in other_elements]
        unique_chains = list(set(binding_chains_raw))

        # 각 체인별 cluster 정보 lookup
        cluster_ids = []
        for chain in unique_chains:
            cluster_id = cluster_info.get((pdb_id, chain))
            print(pdb_id, chain, cluster_id)
            cluster_ids.append(cluster_id)

        results.append([
            pdb_id,
            metal_type,
            [round(coord, 3) for coord in metal_pos.tolist()],
            list(other_elements),
            unique_chains,
            cluster_ids
        ])
    return results


def process_pdb_wrapper(pdb_file):
    """병렬 처리를 위한 Wrapper 함수"""
    print(f"Processing: {pdb_file}")
    return process_pdb(pdb_file)


if __name__ == "__main__":
    # PDB 파일 목록을 텍스트 파일(train_val_test.txt)에서 읽어옴 (각 줄에 pdb_id가 있음)
    with open("/home/qkrgangeun/LigMet/code/text/biolip/filtered/train_val_test.txt", "r") as f:
        pdb_list = [line.strip() for line in f if line.strip()]

    # pdb_list를 기반으로 PDB 파일 경로 생성 (파일명은 pdb_id + '.pdb'라고 가정)
    pdb_files = [pdb_dir / f"{pdb_id}.pdb" for pdb_id in pdb_list]

    # multiprocessing을 사용해 병렬 처리 (예시: 40개 worker)
    with Pool(40) as pool:
        results = pool.map(process_pdb_wrapper, pdb_files)

    # 다중 PDB 파일에 대한 결과를 하나의 리스트로 평탄화
    data = [item for sublist in results for item in sublist]
    print(data)

    # 결과를 CSV 파일로 저장 (헤더에 "Other Elements"로 수정)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["PDB ID", "Metal Type", "Metal Position", "Other Elements", "Binding Chains", "Cluster ID"])
        writer.writerows(data)

    print(f"✅ Metal binding site 정보가 {output_csv}에 저장되었습니다.")
