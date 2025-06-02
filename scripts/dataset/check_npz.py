import os
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# 파일 경로 정의
pdbid_list_path = '/home/qkrgangeun/LigMet/code/text/biolip/paper/train_val_test_filtered.txt'
dl_features_dir = '/home/qkrgangeun/LigMet/data/biolip/dl/features'
rf_features_dir = '/home/qkrgangeun/LigMet/data/biolip/rf/grid_prob'
output_path = 'error_pdbids.txt'  # 결과 저장 파일

# pdbid 목록 읽기
with open(pdbid_list_path, 'r') as f:
    pdbids = [line.strip() for line in f if line.strip()]

# 개별 pdbid 검사 함수
def check_pdbid(pdbid):
    dl_path = os.path.join(dl_features_dir, f'{pdbid}.npz')
    rf_path = os.path.join(rf_features_dir, f'{pdbid}.npz')

    try:
        with np.load(dl_path) as dl_data:
            _ = dl_data['bond_masks']
    except Exception:
        return pdbid

    try:
        with np.load(rf_path) as rf_data:
            _ = rf_data['prob']
    except Exception:
        return pdbid

    return None  # 문제가 없으면 None 반환

# 병렬 처리 + tqdm 진행률 표시
def parallel_check(pdbids):
    error_pdbids = []
    with Pool(cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(check_pdbid, pdbids), total=len(pdbids)):
            if result is not None:
                error_pdbids.append(result)
    return error_pdbids

# 실행
if __name__ == "__main__":
    error_pdbids = parallel_check(pdbids)

    # 에러 pdbid 저장
    with open(output_path, 'w') as f:
        for pdbid in error_pdbids:
            f.write(f"{pdbid}\n")

    print(f"검사 완료: {len(error_pdbids)}개 에러. 결과 저장: {output_path}")
