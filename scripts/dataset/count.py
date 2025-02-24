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
    
    if not metal_positions or not ligand_positions:
        return False
    
    metal_positions = np.array(metal_positions)
    ligand_positions = np.array(ligand_positions)
    
    # 거리 행렬 계산 (모든 금속-리간드 조합 간 거리)
    dists = np.linalg.norm(metal_positions[:, np.newaxis] - ligand_positions, axis=2)
    
    return np.any(dists <= 3.0)

def process_pdb(pdb):
    pdb_dir = Path("/home/qkrgangeun/LigMet/data/pdb")
    if not pdb.endswith(".pdb"):  # 잘못된 파일 필터링
        return False
    
    pdb_id = pdb.replace(".pdb", "")
    pdb_path = pdb_dir / f"{pdb_id}.pdb"
    
    if not pdb_path.exists():  # 다운로드 실패 방지
        print(f"Warning: {pdb_id}.pdb not found in {pdb_dir}")
        return False
    
    return check_metal_ligand_interaction(pdb_path)

def main():
    pdb_dir = Path("/home/qkrgangeun/LigMet/data/pdb")
    pdb_list = [pdb for pdb in os.listdir(pdb_dir) if pdb.endswith(".pdb")]
    
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_pdb, pdb_list)
    
    count = sum(results)
    print(f"Metal-Ligand: {count}")
    print(f"Total: {len(pdb_list)}")

if __name__ == "__main__":
    main()
