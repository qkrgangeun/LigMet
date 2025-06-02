import pandas as pd
import numpy as np
import argparse
from ligmet.utils.rf.rf_features import near_lig, near_res
from ligmet.featurizer import Features
from pathlib import Path

def update_near_lig(input_file, output_file):
    # 기존의 output_file을 읽어옵니다.
    df = pd.read_csv(output_file, compression="gzip")
    
    # npz 파일을 읽어 Features 객체로 변환
    structure_dict = np.load(input_file, allow_pickle=True)
    structure = Features(**structure_dict)
    
    # ligand_mask 설정
    ligand_mask = structure.is_ligand == 1
    dists = [2.5, 2.8, 3.0, 3.2, 5.0]

    # 기존 near_lig 데이터를 새로 계산하여 덮어씁니다.
    for threshold in dists:
        # near_lig 계산
        n_lig_NOS, n_lig_nion, n_lig_etc = near_lig(
            structure.atom_positions[ligand_mask],
            structure.atom_elements[ligand_mask],
            structure.atom_residues[ligand_mask],
            structure.grid_positions,
            threshold
        )
        
        # 새로운 값으로 기존 DataFrame에 덮어씁니다.
        df[f'n_lig_NOS_{threshold}'] = np.array(n_lig_NOS).astype(np.int8)
        df[f'n_lig_nion_{threshold}'] = np.array(n_lig_nion).astype(np.int8)
        df[f'n_lig_etc_{threshold}'] = np.array(n_lig_etc).astype(np.int8)

    # 업데이트된 데이터를 gzip 형식으로 다시 저장합니다.
    df.to_csv(output_file, index=False, compression="gzip")
    
    print(f"Updated data saved to {output_file}")

def main():
    # argparse로 input_path와 output_dir을 받습니다.
    parser = argparse.ArgumentParser(description='Update near_lig data in an existing output file.')
    parser.add_argument('input_path', type=str, help='Path to the npz file (StructureWithGrid)')
    parser.add_argument('--output_dir', type=str, help='Directory for the output file', default=None)
    args = parser.parse_args()

    # input_path와 output_dir 받아오기
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent

    # output_path 설정
    pdb_id = input_path.stem  # .npz 파일 이름에서 확장자 제거
    output_path = output_dir / f'{pdb_id}.csv.gz'

    # near_lig 계산 후 결과 저장
    update_near_lig(input_path, output_path)

if __name__ == "__main__":
    main()
