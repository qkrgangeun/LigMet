import os
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count

# 디렉토리 설정
processed_dir = Path('/home/qkrgangeun/LigMet/data/biolip/processed')
ligand_dir = Path("/home/khs01654/LigScope/data/raw_data/BioLiP_updated_set/ligand")
merge_dir = Path("/home/qkrgangeun/LigMet/data/biolip/merged")
csv_path = "/home/qkrgangeun/LigMet/code/text/biolip/biolip_metal.csv"

# 출력 디렉토리 생성
merge_dir.mkdir(parents=True, exist_ok=True)

# CSV 파일 로드
df = pd.read_csv(csv_path)

# PDB 파일 목록 가져오기
pdb_list = [f for f in os.listdir(processed_dir) if f.endswith(".pdb")]

def merge_pdb(pdb):
    """
    단백질과 리간드를 병합하여 최종 PDB 파일을 생성하는 함수.
    """
    pdb_id = pdb.replace(".pdb", "")
    merged_pdb_path = merge_dir / f"{pdb_id}.pdb"

    # Receptor 파일 (단백질) 병합
    receptor_file = processed_dir / f"{pdb_id}.pdb"
    
    with open(merged_pdb_path, "w") as merged_pdb:
        if receptor_file.exists():
            with open(receptor_file, "r") as r_file:
                for line in r_file:
                    merged_pdb.write(line)

        # Ligand PDB 파일 병합
        ligand_entries = df[df["PDB ID"] == pdb_id]
        for _, row in ligand_entries.iterrows():
            ligand_pdb_path = ligand_dir / f"{row['PDB ID']}_{row['Ligand']}_{row['Ligand chain']}_{row['Ligand Number']}.pdb"
            
            if ligand_pdb_path.exists():
                with open(ligand_pdb_path, "r") as l_file:
                    for line in l_file:
                        if not line.startswith("TER"):  # "TER" 제거
                            merged_pdb.write(line)
            else:
                print('Error',ligand_pdb_path,'not exists')
    print(f"Processed: {merged_pdb_path}")

if __name__ == "__main__":
    num_workers = min(8, cpu_count())  # 최대 8개 프로세스를 사용 (조정 가능)
    with Pool(num_workers) as pool:
        pool.map(merge_pdb, pdb_list)
