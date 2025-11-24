import numpy as np
from pathlib import Path
from ligmet.utils.pdb import read_pdb # type: ignore
from ligmet.featurizer import * # type: ignore
from ligmet.utils.grid import * # type: ignore
from dataclasses import asdict
import traceback
import argparse
# # 입력 및 출력 디렉토리 설정
# pdb_dir = Path('/home/qkrgangeun/LigMet/data/biolip/pdb')  # PDB 파일이 있는 디렉토리
# output_dir = Path('/home/qkrgangeun/LigMet/data/biolip/dl/features')  # .npz 저장할 디렉토리
# #metalpred
# # pdb_dir = Path('/home/qkrgangeun/MetalPred/data/biolip_group/latest')  # PDB 파일이 있는 디렉토리
# # output_dir = Path('/home/qkrgangeun/LigMet/data/metalpred/dl/features')  
# #af3
# pdb_dir = Path('/home/qkrgangeun/LigMet/data/biolip_backup/test')
# output_dir = Path('/home/qkrgangeun/LigMet/data/biolip_backup/test')
# output_dir.mkdir(parents=True, exist_ok=True)  # 저장 폴더 생성 (없으면 생성)

def bondmask_to_neighidx(bond_mask: np.ndarray):
    rows, cols = np.where(np.triu(bond_mask) > 0)
    return np.stack([rows, cols], axis=0).astype(np.int32)

def optimize_dtype(key, arr):
    if key == "bond_masks":
        return bondmask_to_neighidx(arr)

    elif key in ["atom_positions", "metal_positions", "grid_positions", "qs", "sasas"]:
        return arr.astype(np.float32)

    elif key == "residue_idxs":
        return arr.astype(np.int32)

    elif key in ["sec_structs", "gen_types"]:
        return arr.astype(np.int16)

    elif key == "is_ligand":
        return arr.astype(np.bool_)

    elif isinstance(arr, np.ndarray) and arr.dtype.kind == "U":
        maxlen = max(len(str(s)) for s in arr)
        return arr.astype(f"<U{maxlen}")

    return arr  # 그대로

def process_pdb(pdb_path , output_dir):
    """ 개별 PDB 파일을 처리하고 .npz로 저장하는 함수 """
    pdb_path = Path(pdb_path)
    pdb_id = pdb_path.stem
    output_npz_path = f"{output_dir}/{pdb_id}.npz"
    # if output_npz_path.exists():
    #     print(f"already exit Skip: {pdb_path.name}")
    #     return
    try:
        print(f"📂 Processing: {pdb_path.name}")

        # PDB 데이터 읽기
        structure = read_pdb(pdb_path)
        if len(structure.atom_positions) > 50000:
            print("skip more than 50000")
            return
        else:
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

            # `.npz` 파일로 저장할 경로 설정

            # 저장할 데이터를 딕셔너리로 변환
            feature_dict = {
                k: optimize_dtype(k, v)
                for k, v in asdict(features).items()
                if isinstance(v, np.ndarray)
            }

            # `.npz` 파일로 압축 저장
            np.savez(output_npz_path, **feature_dict)

            print(f"✅ {pdb_path.name} 처리 완료 → 저장됨: {output_npz_path}")

    except Exception as e:
        print(f"❌ {pdb_path.name} 처리 중 오류 발생: {e}")
        traceback.print_exc()  # 상세 오류 출력

# 병렬 처리 실행
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('pdb_path', type=str )
    parser.add_argument('--output_dir', type=str, default='.') 
    args = parser.parse_args()
    pdb_path = args.pdb_path
    output_dir = args.output_dir
    process_pdb(pdb_path, output_dir)

