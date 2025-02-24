import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
from Bio import PDB
import numpy as np

def check_metal_ligand_interaction(file):
    metals = {"MG", "ZN", "MN", "CA", "FE", "NI", "CO", "CU", "K", "NA"}
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("structure", file)
    
    metal_positions = []
    ligand_positions = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                res_name = residue.get_resname().strip()
                if "H_" in residue.id[0]: 
                    for atom in residue:
                        atom_name = atom.element.strip().upper()
                        coord = atom.get_coord()
                        if atom_name in metals:
                            metal_positions.append(coord)
                        elif res_name != "HOH":
                            ligand_positions.append(coord)
    
    if not ligand_positions:
        return 0, len(metal_positions)
    
    metal_positions = np.array(metal_positions)
    ligand_positions = np.array(ligand_positions)
    
    # 거리 행렬 계산 (모든 금속-리간드 조합 간 거리)
    dists = np.linalg.norm(metal_positions[:, np.newaxis] - ligand_positions, axis=2)
    bound_metals = np.sum(np.any(dists < 3.0, axis=-1))
    return bound_metals, len(metal_positions)

def process_pdb(pdb):
    pdb_dir = Path("/home/qkrgangeun/LigMet/data/pdb")
    pdb_path = pdb_dir / pdb
    if not pdb_path.exists():  # 다운로드 실패 방지
        print(f"Warning: {pdb} not found in {pdb_dir}")
        return 0, 0
    return check_metal_ligand_interaction(pdb_path)

def main():
    pdb_dir = Path("/home/qkrgangeun/LigMet/data/pdb")
    pdb_list = [pdb for pdb in os.listdir(pdb_dir) if pdb.endswith(".pdb")]
    
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_pdb, pdb_list)
    
    total_bound = sum(r[0] for r in results)
    total_metals = sum(r[1] for r in results)
    
    print("Metal-Ligand:", total_bound)
    print("Total:", total_metals)

if __name__ == "__main__":
    main()