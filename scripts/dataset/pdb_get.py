import os
import subprocess
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count

# 경로 설정
csv_path = "/home/qkrgangeun/LigMet/code/text/biolip/biolip_metal.csv"
pdb_dir = "/home/qkrgangeun/LigMet/data/biolip/pdb"
os.makedirs(pdb_dir, exist_ok=True)

# CSV 파일 로드
df = pd.read_csv(csv_path)

# Unique PDB ID 리스트 생성
unique_pdb_ids = df["PDB ID"].unique()

# PDB 디렉토리 이동
os.chdir(pdb_dir)

# 병렬 다운로드 함수 정의
def download_pdb(pdb_id: str):
    """pdb_get 명령어를 실행하여 PDB 파일을 다운로드하는 함수"""
    try:
        subprocess.run(["pdb_get", pdb_id], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {pdb_id}: {e}")

# 병렬 처리 설정
num_workers = min(8, cpu_count())  # 최대 8개 프로세스를 사용 (조정 가능)
if __name__ == "__main__":
    with Pool(num_workers) as pool:
        pool.map(download_pdb, unique_pdb_ids)
