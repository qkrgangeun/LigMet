import sys
import numpy as np
from pathlib import Path
from ligmet.utils.pdb import read_pdb
from ligmet.featurizer import *
from ligmet.utils.grid import *
from dataclasses import asdict
import traceback

# 입력 및 출력 디렉토리 설정
pdb_dir = Path('/home/qkrgangeun/LigMet/data/biolip/pdb')  # PDB 파일이 있는 디렉토리
output_dir = Path('/home/qkrgangeun/LigMet/data/biolip/dl/features')  # .npz 저장할 디렉토리
output_dir.mkdir(parents=True, exist_ok=True)  # 저장 폴더 생성 (없으면 생성)

# 모든 PDB 파일 리스트
total_files = list(pdb_dir.glob("*.pdb"))

# 시작과 끝 인덱스 인자 받기
start_idx = int(sys.argv[1])
end_idx = int(sys.argv[2])
pdb_files = total_files[start_idx:end_idx]

print(f"🔹 총 {len(pdb_files)}개의 PDB 파일을 처리합니다. (Index {start_idx} ~ {end_idx})")

def process_pdb(pdb_path):
    """ 개별 PDB 파일을 처리하고 .npz로 저장하는 함수 """
    try:
        output_npz_path = output_dir / f"{pdb_path.stem}.npz"
        
        # 이미 존재하는 경우 스킵
        if output_npz_path.exists():
            print(f"⏩ {pdb_path.name} 이미 존재하여 스킵합니다.")
            return
        
        print(f"📂 Processing: {pdb_path.name}")

        # PDB 데이터 읽기
        structure = read_pdb(pdb_path)
        if len(structure.atom_positions) > 90000:
            print(f"⏩ {pdb_path.name} 90000개 초과하여 스킵합니다.")
            return
        # Grid 생성 및 필터링
        grids = sasa_grids_thread(structure.atom_positions, structure.atom_elements)
        grids = filter_by_clashmap(grids)

        # StructureWithGrid 생성
        structure_dict = asdict(structure)
        structure_with_grid = StructureWithGrid(
            grid_positions=grids,
            **structure_dict
        )

    # Features 생성
        features = make_features(pdb_path, structure_with_grid)

        # 저장할 데이터를 딕셔너리로 변환
        feature_dict = {k: v for k, v in asdict(features).items() if isinstance(v, np.ndarray)}

        # `.npz` 파일로 압축 저장
        np.savez_compressed(output_npz_path, **feature_dict)

        print(f"✅ {pdb_path.name} 처리 완료 → 저장됨: {output_npz_path}")

    except Exception as e:
        print(f"❌ {pdb_path.name} 처리 중 오류 발생: {e}")
        traceback.print_exc()  # 상세 오류 출력

# 지정된 범위의 파일을 처리
if __name__ == "__main__":
    for pdb_file in pdb_files:
        process_pdb(pdb_file)

    print("🎉 지정된 범위의 PDB 파일 처리가 완료되었습니다!")
