import os
from pathlib import Path
import sys
from multiprocessing import Pool, cpu_count

# renumbering 함수가 정의된 경로를 모듈로 추가
sys.path.append('/home/qkrgangeun/LigMet/code/scripts/dataset')
from pdb_renum import renumbering

# 경로 설정
af2_dir = '/home/qkrgangeun/LigMet/data/biolip/af2.3/DB'
orig_pdb_dir = '/home/qkrgangeun/LigMet/data/biolip/pdb'
output_dir = '/home/qkrgangeun/LigMet/data/biolip/renum_pdb'

# 작업 단위 함수 정의
def process_pdb(file):
    if not (file.startswith('AF_') and file.endswith('.pdb')):
        return

    pdbid = file[3:7].lower()
    orig_pdb_path = os.path.join(orig_pdb_dir, f'{pdbid}.pdb')
    af2_pdb_path = os.path.join(af2_dir, file)
    output_pdb_path = os.path.join(output_dir, f'{pdbid}.pdb')

    if not os.path.exists(orig_pdb_path):
        return f'SKIPPED (missing original): {pdbid}'

    try:
        renumbering(orig_pdb_path, af2_pdb_path, 'A', output_pdb_path)
        return f'DONE: {pdbid}'
    except Exception as e:
        return f'ERROR ({pdbid}): {e}'

if __name__ == '__main__':
    files = os.listdir(af2_dir)

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_pdb, files)
