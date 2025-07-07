#!/usr/bin/env python
# dl_features.py

import argparse
import traceback
from pathlib import Path
import numpy as np

from ligmet.utils.pdb import read_pdb  # type: ignore
from ligmet.featurizer import make_features  # type: ignore
from ligmet.utils.grid import sasa_grids_thread, filter_by_clashmap  # type: ignore
from dataclasses import asdict

def process_pdb(pdb_path: Path, output_dir: Path):
    """
    개별 PDB 파일을 처리하고 .npz로 저장하는 함수.
    - pdb_path: .pdb 파일 경로
    - output_dir: .npz 저장 디렉토리
    """
    pdb_id = pdb_path.stem
    output_npz = output_dir / f"{pdb_id}.npz"
    try:
        print(f"📂 Processing: {pdb_path.name}")

        structure = read_pdb(pdb_path)
        if len(structure.atom_positions) > 50000:
            print("⚠️  skip: >50000 atoms")
            return

        # Grid 생성 및 필터링
        grids = sasa_grids_thread(structure.atom_positions, structure.atom_elements)
        grids = filter_by_clashmap(grids)

        # StructureWithGrid 생성
        structure_dict = asdict(structure)
        # ↓ 여기가 추가된 부분: metal_* 항목이 없으면 빈 배열로 채워주기
        import numpy as np
        structure_dict.setdefault("metal_positions", np.empty((0,3)))
        structure_dict.setdefault("metal_types",    np.empty((0,), dtype=object))

        from ligmet.utils.pdb import StructureWithGrid  # type: ignore
        structure_with_grid = StructureWithGrid(
            grid_positions=grids,
            **structure_dict
        )

        # Features 생성
        features = make_features(pdb_path, structure_with_grid)

        # `.npz` 파일로 저장할 데이터 준비
        feature_dict = {
            k: v
            for k, v in asdict(features).items()
            if isinstance(v, np.ndarray)
        }

        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(output_npz, **feature_dict)
        print(f"✅ Saved: {output_npz}")

    except Exception as e:
        print(f"❌ Error processing {pdb_path.name}: {e}")
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Compute DL features for a PDB and save as .npz"
    )
    parser.add_argument(
        "pdb_input",
        type=Path,
        help="Path to a .pdb file, or a directory containing multiple .pdb files",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=Path,
        required=True,
        help="Directory where .npz feature files will be written",
    )
    args = parser.parse_args()

    # 입력이 디렉토리면 그 안의 모든 .pdb를 처리
    if args.pdb_input.is_dir():
        pdb_files = sorted(args.pdb_input.glob("*.pdb"))
    else:
        pdb_files = [args.pdb_input]

    for pdb_path in pdb_files:
        process_pdb(pdb_path, args.output_dir)

if __name__ == "__main__":
    main()
