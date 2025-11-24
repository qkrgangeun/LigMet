import numpy as np
from pathlib import Path
from ligmet.utils.pdb import read_pdb # type: ignore
from ligmet.featurizer import * # type: ignore
from ligmet.utils.grid import * # type: ignore
from dataclasses import asdict
import traceback
import argparse

class PruneDataSet():
    def__init__(self, pdb_dir: str, output_dir: str, skip_existing: bool = True):
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
        
from ligmet.utils.rf.rf_features import near_lig, near_res, nearest_bb_dist, nearest_prot_carbon_dist, binned_res, parse_pdb, filter_by_biometall, RSA # type: ignore
import numpy as np
import pandas as pd
import argparse
from ligmet.utils.pdb import StructureWithGrid # type: ignore
from ligmet.utils.constants import aliphatic_carbons, aromatic_carbons # type: ignore
from ligmet.utils.rf.label import label_grids # type: ignore
from ligmet.featurizer import Features # type: ignore
from pathlib import Path
def process_file(file_path, output_file):
    structure_dict = np.load(file_path, allow_pickle=True)
    structure = Features(**structure_dict)
    
    protein_mask = structure.is_ligand == 0
    ligand_mask = structure.is_ligand == 1
    dists = [2.5, 2.8, 3.0, 3.2, 5.0]
    df = pd.DataFrame()
    
    for threshold in dists:
        p_coords_res, p_core_res, p_bb_coords = near_res(
            structure.atom_residues[protein_mask],
            structure.atom_names[protein_mask],
            structure.atom_elements[protein_mask],
            structure.atom_positions[protein_mask],
            structure.grid_positions,
            threshold
        )
        
        n_lig_NOS, n_lig_nion, n_lig_etc = near_lig(
            structure.atom_positions[ligand_mask],
            structure.atom_elements[ligand_mask],
            structure.atom_residues[ligand_mask],
            structure.grid_positions,
            threshold
        )
        
        df[f'p_coords_res_{threshold}'] = np.array(p_coords_res).astype(np.int8)
        df[f'p_core_res_{threshold}'] = np.array(p_core_res).astype(np.int8)
        df[f'p_bb_coords_{threshold}'] = np.array(p_bb_coords).astype(np.int8)
        df[f'n_lig_NOS_{threshold}'] = np.array(n_lig_NOS).astype(np.int8)
        df[f'n_lig_nion_{threshold}'] = np.array(n_lig_nion).astype(np.int8)
        df[f'n_lig_etc_{threshold}'] = np.array(n_lig_etc).astype(np.int8)
    
    n_coords_res_bin, n_core_res_bin, n_bb_coords_bin = binned_res(
        structure.atom_residues,
        structure.atom_names,
        structure.atom_elements,
        structure.atom_positions,
        structure.grid_positions,
        3,
        5
    )
    
    df['n_coords_res_bin'] = np.array(n_coords_res_bin).astype(np.int8)
    df['n_core_res_bin'] = np.array(n_core_res_bin).astype(np.int8)
    df['n_bb_coords_bin'] = np.array(n_bb_coords_bin).astype(np.int8)
    
    min_dist = nearest_prot_carbon_dist(
        structure.atom_residues,
        structure.atom_names,
        structure.atom_elements,
        structure.atom_positions,
        structure.grid_positions,
        aliphatic_carbons,
        aromatic_carbons
    )
    df['min_c_dist'] = np.array(min_dist).astype(np.float16)
    
    near_bb_dist_values = nearest_bb_dist(
        structure.atom_names[protein_mask],
        structure.atom_positions[protein_mask],
        structure.grid_positions
    )
    df['near_bb_dist'] = np.array(near_bb_dist_values).astype(np.float16)
    
    sasa = RSA(structure.grid_positions, structure.atom_positions, structure.atom_elements)
    df['sasa'] = np.array(sasa).astype(np.float16)
    
    atom_dict, grids = parse_pdb(structure)
    num_res_list = filter_by_biometall(grids, atom_dict)
    df['biometall'] = np.array(num_res_list).astype(np.int8)
    
    #### Label ####
    labels = label_grids(structure.metal_positions, structure.grid_positions, 2.0)
    df['label_2.0'] = labels.astype('bool')
    
    df.to_csv(output_file, index=False, compression="gzip")
    
    import argparse
from joblib import load
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import os
from pathlib import Path
def load_data(pdb_id, label_column='label_2.0'):
    """테스트 데이터 로드 및 전처리"""
    data_path = pdb_id
    data = pd.read_csv(data_path, compression='gzip')
    X = data.drop([label_column], axis=1)
    Y = data[label_column]
    return X, Y

def save_predicted_grids(pdb_id, y_pred, output_dir):
    """True로 예측된 grid의 위치를 .xyz 파일로 저장"""
    npz_path = f"/home/qkrgangeun/LigMet/data/biolip/dl/features/{pdb_id}.npz"
    npz_path = f"/home/qkrgangeun/LigMet/data/biolip_backup/af2.3/testset_chain1/dl/features/{pdb_id}.npz"
    npz_path = "./"
    df = np.load(npz_path)
    grid_positions = df["grid_positions"]
    true_grid_positions = grid_positions[y_pred == 1]

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{pdb_id}_grid.xyz")

    with open(output_path, 'w') as file:
        for i, coord in enumerate(true_grid_positions, start=1):
            # PDB ATOM 레코드 형식: ATOM, serial, name, resName, chainID, resSeq, x, y, z, occupancy, tempFactor, element
            line = (
                f"ATOM  {i:5d}  H   LIG A   1    "
                f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                f"  1.00  0.00           H\n"
            )
            file.write(line)
    print(f"Predicted grid saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, default="/home/qkrgangeun/LigMet/data/rf_param/0415_newlabel_chain1", help="Path to the trained model file (.joblib)")
    parser.add_argument("--test_data", type=str, required=True, help="randomforest feature csv path: {pdbid}.csv.gz")
    parser.add_argument("--output_dir", type=str, default='/home/qkrgangeun/LigMet/data/biolip/rf/grid_prob', help="Output directory for prediction")

    args = parser.parse_args()
    input_path = args.test_data
    pdb_id = Path(input_path).with_suffix("").stem
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{pdb_id}.npz"

    # 이미 출력 파일이 존재하면 건너뜀
    # if output_path.exists():
    #     print(f"⏭️  Skip (Already Exists): {output_path}")
    #     return

    # 모델과 테스트 데이터 로딩
    print("-> Loading model and test data")
    model = load(args.model_path)
    X_test, Y_test = load_data(input_path)

    # 예측 수행
    y_prob = model.predict_proba(X_test)[:, 1]

    # 결과 저장
    np.savez(output_path, prob=y_prob)
    print(f"✅ Prediction complete → Saved to: {output_path}")
