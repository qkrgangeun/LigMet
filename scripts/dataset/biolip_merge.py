import os
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count

def merge_pdb_files(pdb_id, receptor_dir, ligand_dir, merge_dir, df):
    merged_pdb_path = merge_dir / f"{pdb_id}.pdb"
    
    # Receptor 파일들 찾기 (예: 1a0dA.pdb, 1a0dB.pdb 등)
    receptor_files = list(receptor_dir.glob(f"{pdb_id}*.pdb"))
    
    with open(merged_pdb_path, "w") as merged_pdb:
        # Receptor PDB 파일들 병합
        for receptor_file in receptor_files:
            with open(receptor_file, "r") as r_file:
                for line in r_file:
                    if not line.startswith("TER"):  # "TER" 제거
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
    
    print(f"Merged: {merged_pdb_path}")

def main():
    # 디렉토리 설정
    receptor_dir = Path("/home/khs01654/LigScope/data/raw_data/BioLiP_updated_set/receptor")
    ligand_dir = Path("/home/khs01654/LigScope/data/raw_data/BioLiP_updated_set/ligand")
    merge_dir = Path("/home/qkrgangeun/LigMet/data/biolip/merged")
    csv_path = "/home/qkrgangeun/LigMet/code/text/biolip/biolip_metal.csv"

    # 출력 디렉토리 생성
    merge_dir.mkdir(parents=True, exist_ok=True)

    # CSV 파일 로드
    df = pd.read_csv(csv_path)

    # Unique PDB ID 리스트 생성
    unique_pdb_ids = df["PDB ID"].unique()

    # 멀티프로세싱 실행
    with Pool(processes=cpu_count()) as pool:
        pool.starmap(merge_pdb_files, [(pdb_id, receptor_dir, ligand_dir, merge_dir, df) for pdb_id in unique_pdb_ids])

if __name__ == "__main__":
    main()