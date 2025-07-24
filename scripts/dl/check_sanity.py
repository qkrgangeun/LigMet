#!/usr/bin/env python3
import os
import numpy as np
import multiprocessing

# 설정
pdb_list_file = "/home/qkrgangeun/LigMet/code/text/biolip/small_train_pdbs.txt"
npz_dir = "/home/qkrgangeun/LigMet/data/biolip/dl/features"
broken_files_log = "/home/qkrgangeun/LigMet/code/text/biolip/broken_npz_files.txt"

# CPU 코어 수 설정
NUM_WORKERS = min(8, multiprocessing.cpu_count())  # 최대 8개 워커 사용 (조정 가능)


def check_npz_integrity(pdb_id):
    """주어진 PDB ID에 해당하는 npz 파일이 정상적인지 확인"""
    npz_file = os.path.join(npz_dir, f"{pdb_id}.npz")

    if not os.path.exists(npz_file):
        return f"File not found: {npz_file}"

    try:
        with np.load(npz_file, allow_pickle=True) as data:
            if len(data.files) == 0:
                raise ValueError("Empty NPZ file")
        return None  # 정상적인 경우 아무것도 반환하지 않음
    except Exception as e:
        return f"Broken file: {npz_file} - {e}"


def main():
    """멀티프로세싱을 사용하여 npz 무결성 검사"""
    with open(pdb_list_file, "r") as f:
        pdb_ids = [line.strip() for line in f if line.strip()]

    # 멀티프로세싱 풀 생성
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(check_npz_integrity, pdb_ids)

    # 오류가 발생한 파일만 필터링 (None 값 제거)
    broken_files = [r for r in results if r is not None]

    # 로그 저장
    if broken_files:
        with open(broken_files_log, "w") as log_file:
            log_file.write("\n".join(broken_files) + "\n")
        print(f"Broken NPZ files are logged in: {broken_files_log}")

    # 오류가 발생한 파일만 출력
    for res in broken_files:
        print(res)


if __name__ == "__main__":
    main()
