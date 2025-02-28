import os
from collections import defaultdict
from multiprocessing import Pool, cpu_count

# PDB 파일이 저장된 경로
pdb_dir = "/home/qkrgangeun/LigMet/data/biolip/pdb"

# metals 리스트 (금속 원소들)
metals = {"MG", "ZN", "MN", "CA", "FE", "NI", "CO", "CU", "K", "NA"}

# PDB 리스트 파일 경로
train_txt_path = "/home/qkrgangeun/LigMet/code/text/biolip/train_pdbs.txt"
val_txt_path = "/home/qkrgangeun/LigMet/code/text/biolip/val_pdbs.txt"
test_txt_path = "/home/qkrgangeun/LigMet/code/text/biolip/test_pdbs.txt"

# 파일 읽기
def load_pdb_list(file_path):
    with open(file_path, "r") as f:
        return [line.strip() for line in f]

train_pdbs = load_pdb_list(train_txt_path)
val_pdbs = load_pdb_list(val_txt_path)
test_pdbs = load_pdb_list(test_txt_path)

# 금속 개수 세기 함수 (PDB 하나씩 처리)
def count_metals_in_pdb(pdb_id):
    pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
    metal_counts = defaultdict(int)

    if not os.path.exists(pdb_path):
        return metal_counts

    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("HETATM"):
                atom = line[12:17].strip()
                res = line[18:21].strip()
                if atom in metals:
                    metal_counts[atom] += 1
                    print(atom)
    return metal_counts

# 병렬 처리 함수
def count_metals_parallel(pdb_list):
    metal_counts = defaultdict(int)
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(count_metals_in_pdb, pdb_list)

    # 결과 합산
    for result in results:
        for metal, count in result.items():
            metal_counts[metal] += count

    return metal_counts

# 각 데이터셋에서 금속 종류별 개수 계산 (멀티프로세싱 적용)
train_metal_counts = count_metals_parallel(train_pdbs)
val_metal_counts = count_metals_parallel(val_pdbs)
test_metal_counts = count_metals_parallel(test_pdbs)

# 결과 출력
def print_metal_counts(dataset_name, metal_counts):
    print(f"\n{dataset_name} 데이터에서 금속 종류별 개수:")
    for metal, count in sorted(metal_counts.items()):
        print(f"  {metal}: {count}")

print_metal_counts("Training", train_metal_counts)
print_metal_counts("Validation", val_metal_counts)
print_metal_counts("Test", test_metal_counts)
