import csv
import os
from pathlib import Path
from scipy.spatial import cKDTree
from multiprocessing import Pool
from ligmet.utils.pdb import read_pdb
import numpy as np

# Metal 원소 리스트
metals = {"ZN", "MG", "FE", "CA", "CU", "MN", "CO", "NI", "NA", "K"}

# PDB 파일이 저장된 디렉토리
pdb_dir = Path("/home/qkrgangeun/LigMet/data/biolip/merged")  # 실제 PDB 파일 경로로 변경
output_csv = "/home/qkrgangeun/LigMet/code/text/biolip/metal_binding_sites3.csv"
cluster_csv = "/home/qkrgangeun/LigMet/code/text/biolip/clusterid_releaseddate.csv"
# filtered_pdbs.csv에서 pdb_id, chain_id, cluster 정보를 읽어오는 함수
def read_cluster_info(cluster_csv):
    cluster_info = {}
    with open(cluster_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdb_id = row['pdb_id']
            chains = eval(row['chains'])  # 문자열을 리스트로 변환
            clusters = eval(row['clusters'])  # 문자열을 리스트로 변환
            for chain, cluster in zip(chains, clusters):
                cluster_info[(pdb_id, chain)] = cluster
    return cluster_info

# cluster 정보를 읽어옴
cluster_info = read_cluster_info(cluster_csv)

def process_pdb(pdb_file, cutoff=3.0):
    """주어진 PDB 파일에서 Metal Binding Residues를 찾아 반환"""
    structure = read_pdb(pdb_file)

    # Metal 원자가 없는 경우 건너뜀
    if structure.metal_positions is None or len(structure.metal_positions) == 0:
        print('no metal')
        return []

    # NumPy 배열로 변환
    metal_positions = np.array(structure.metal_positions)  # (M, 3)
    metal_types = np.array(structure.metal_types)  # (M,)
    atom_positions = structure.atom_positions  # (N, 3)
    atom_residues = structure.atom_residues  # (N,)
    residue_idxs = structure.residue_idxs  # (N,)
    chain_ids = structure.chain_ids
    # KDTree를 사용하여 Metal 원자의 주변 원자 찾기
    tree = cKDTree(atom_positions)
    neigh_indices = tree.query_ball_point(metal_positions, cutoff)  # 리스트 (M x variable)
    print(metal_positions)
    print(neigh_indices)
    # 결과 저장용 리스트
    results = []
    for metal_idx, metal_pos in enumerate(metal_positions):
        metal_type = metal_types[metal_idx]

        # 현재 Metal 주변 3Å 이내의 원자 인덱스 추출
        if len(neigh_indices[metal_idx]) > 0:
            neigh_residues = atom_residues[neigh_indices[metal_idx]]
            neigh_residue_idxs = residue_idxs[neigh_indices[metal_idx]]

            # Residue 중복 제거
            unique_chains = list(set(chain_ids[neigh_indices[metal_idx]]))
            unique_residues = set(zip(neigh_residues, neigh_residue_idxs))
            binding_residues = [res for res, _ in unique_residues]  # Residue 이름만 반환
            print(unique_chains)
            print(unique_residues)
            # cluster 정보 추가
            pdb_id = pdb_file.stem
            cluster_ids = []
            for chain in unique_chains:
                cluster_id = cluster_info.get((pdb_id, chain))
                print(pdb_id, chain ,cluster_id)
                    # 결과 저장 (metal_pos는 소수점 3째자리까지 반올림)
                cluster_ids.append(cluster_id)
            results.append([pdb_id, metal_type, 
                            [round(coord, 3) for coord in metal_pos.tolist()],  # Metal position 소수점 3자리로 반올림
                            binding_residues, unique_chains, cluster_ids])

    return results

def process_pdb_wrapper(pdb_file):
    """병렬 처리를 위한 Wrapper 함수"""
    print(f"Processing: {pdb_file}")
    return process_pdb(pdb_file)

if __name__ == "__main__":
    # PDB 파일 리스트 가져오기
    pdb_files = [file.strip() for file in open("/home/qkrgangeun/LigMet/code/text/biolip/filtered/train_val_test.txt")]
    pdb_files = [pdb_dir / f"{file.strip()}.pdb" for file in open("/home/qkrgangeun/LigMet/code/text/biolip/filtered/train_val_test.txt")]
    # 병렬 처리 실행 (최대 CPU 개수 사용)
    num_workers = os.cpu_count()  # 가용 CPU 개수
    with Pool(40) as pool:
        results = pool.map(process_pdb_wrapper, pdb_files)

    # 결과를 한 개의 리스트로 합침
    data = [item for sublist in results for item in sublist]
    print(data)
    # CSV 파일로 저장
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["PDB ID", "Metal Type", "Metal Position", "Binding Residues", "Binding Chains", "Cluster ID"])
        writer.writerows(data)

    print(f"✅ Metal binding site 정보가 {output_csv}에 저장되었습니다.")
