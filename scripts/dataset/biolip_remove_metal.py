import os
from pathlib import Path
from multiprocessing import Pool, cpu_count

# 경로 설정
pdb_dir = "/home/qkrgangeun/LigMet/data/biolip/pdb"
processed_dir = Path("/home/qkrgangeun/LigMet/data/biolip/processed")

# processed_dir이 존재하지 않으면 생성
processed_dir.mkdir(parents=True, exist_ok=True)

# 금속 원소 리스트 (주기율표에 기반)
METALS = {
    "LI", "NA", "K", "RB", "CS", "FR",
    "BE", "MG", "CA", "SR", "BA", "RA",
    "SC", "Y", "TI", "ZR", "HF", "RF",
    "V", "NB", "TA", "DB", "CR", "MO", "W", "SG",
    "MN", "TC", "RE", "BH", "FE", "RU", "OS", "HS",
    "CO", "RH", "IR", "MT", "NI", "PD", "PT", "DS",
    "CU", "AG", "AU", "RG", "ZN", "CD", "HG", "CN",
    "AL", "GA", "IN", "TL", "SN", "PB", "BI", "PO"
}

# 물(Water) 잔기 리스트
WATER_RESIDUES = {"HOH", "WAT", "H2O"}

# PDB 파일 목록 가져오기
pdb_list = [f for f in os.listdir(pdb_dir) if f.endswith(".pdb")]

def process_pdb(file):
    """ 하나의 PDB 파일을 처리하여 Metal과 Water를 제외한 Ligand만 저장하는 함수 """
    pdb_id = file.replace(".pdb", "")
    input_path = os.path.join(pdb_dir, file)
    output_path = processed_dir / f"{pdb_id}.pdb"

    with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            if line.startswith("ATOM"):
                f_out.write(line)
            elif line.startswith("HETATM"):
                residue_name = line[17:20].strip()  # Residue Name
                atom_name = line[12:16].strip()  # Atom Name

                # Metal 이온 필터링 (residue_name 또는 atom_name이 metal 리스트에 포함)
                if atom_name in METALS and atom_name in residue_name:
                    continue  # metal이면 무시
                
                # Water 필터링
                if residue_name in WATER_RESIDUES:
                    continue  # 물이면 무시

                # Metal, Water가 아닌 Ligand만 저장
                f_out.write(line)

    print(f"Processed: {output_path}")

if __name__ == "__main__":
    num_workers = min(8, cpu_count())  # 최대 8개 프로세스를 사용 (조정 가능)
    with Pool(num_workers) as pool:
        pool.map(process_pdb, pdb_list)
