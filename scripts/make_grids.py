from ligmet.utils.grid import sasa_grids, filter_by_clashmap
from ligmet.utils.pdb import read_pdb
import os
from dataclasses import asdict
import numpy as np
import argparse

def main(pdb_path, save_dir):
    try:
        structure = read_pdb(pdb_path)
        grids = sasa_grids(structure.atom_positions, structure.atom_elements)
        grids = filter_by_clashmap(grids)
        
        if len(grids) > 0:

            prefix = pdb_path.split("/")[-1].split(".pdb")[0]
            
            # dataclass 객체를 딕셔너리로 변환
            structure_dict = asdict(structure)
            data_dict = {
                'grid_positions': grids,
                **structure_dict  # structure_dict의 내용을 추가
            }
 
            np.savez(
                os.path.join(save_dir, f'{prefix}.npz'),
                **data_dict  # data_dict의 키와 값을 unpack하여 저장
            )
            print(os.path.join(save_dir, f'{prefix}.npz'))
            
            with open(f"{save_dir}/{prefix}_grids.pdb", "w") as file:
                for atom_idx, xyz in enumerate(grids, start=1):
                    file.write(
                        f"HETATM{atom_idx:5}  GRD GRD A{atom_idx:4}    "
                        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  0.00  1.00           H\n"
                    )
    except Exception as e:
        print(f"Error: {e}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PDB files.")
    parser.add_argument("-f", type=str)
    parser.add_argument(
        "-pdb_dir",
        type=str,
        default="/home/qkrgangeun/simple_DLcode3/biolip/data/nonredund_combined",
    )
    parser.add_argument(
        "-save_dir", type=str, default="/home/qkrgangeun/LigMet/data/grids"
    )

    args = parser.parse_args()
    pdb_file = args.f
    pdb_dir = args.pdb_dir
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    if pdb_file.endswith(".pdb"):
        pdb_path = os.path.join(pdb_dir, pdb_file)
        print(pdb_path)
        main(pdb_path, save_dir)
    else:
        pdbs = [line.strip() for line in open(pdb_file)]
        for pdb in pdbs:
            # to ensure pdb_id.pdb
            pdb = pdb.split(".pdb")[0] + ".pdb"
            pdb_path = os.path.join(pdb_dir, pdb)
            print(pdb_path)
            main(pdb_path, save_dir)