from ligmet.utils.rf_features import near_lig, near_res, nearest_bb_dist, nearest_prot_carbon_dist, binned_res, parse_pdb, filter_by_biometall, RSA
import numpy as np
import pandas as pd
import argparse
from ligmet.utils.pdb import StructureWithGrid
from ligmet.utils.constants import aliphatic_carbons, aromatic_carbons
from ligmet.utils.rf.label import label_grids
from pathlib import Path
def process_file(file_path, output_file):
    structure_dict = np.load(file_path, allow_pickle=True)
    structure = StructureWithGrid(**structure_dict)
    
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
    
    df.to_csv(output_file, index=False)
def main():
    parser = argparse.ArgumentParser(description='Process an npz file and extract features.')
    parser.add_argument('input_path', type=str, help='Path to the npz file, StructureWithGrid')
    parser.add_argument('--output_dir', type=str, help='Path to the output directory')
    args = parser.parse_args()
    
    input_path = Path(args.input_path)  # 'input_path'로 수정
    file_dir, file_name = input_path.parent, input_path.name
    pdb_id = file_name.split('.npz')[0]
    
    # output_dir이 args.output_dir이 아니면 default 'rf_features' 디렉터리로 설정
    output_dir = Path(args.output_dir) if args.output_dir else file_dir.parent / 'rf_features'
    output_path = output_dir / f'{pdb_id}.csv'
    output_dir.mkdir(parents=True, exist_ok=True)
    process_file(input_path, output_path)
    
    print("Processing complete. Data saved to", output_path)
if __name__ == "__main__":
    main()
