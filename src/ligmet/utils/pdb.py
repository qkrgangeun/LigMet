from dataclasses import dataclass
import numpy as np
import io
from collections import defaultdict
from Bio.PDB.PDBParser import PDBParser
from ligmet.utils.constants import metals

@dataclass
class Structure:
    atom_positions: np.ndarray  # [n_atoms, 3]
    atom_names: np.ndarray  # [n_atoms, 1]
    atom_elements: np.ndarray  # [n_atoms, 1]
    atom_residues: np.ndarray  # [n_atoms, 1] if ligand: x
    residue_idxs: np.ndarray #[n_atoms, 1]
    chain_ids: np.ndarray #[n_atoms,1]
    is_ligand: np.ndarray  # [n_atoms, 1]
    metal_positions: np.ndarray  # [n_metals, 3]
    metal_types: np.ndarray  # [n_metals, 1]

@dataclass
class StructureWithGrid:
    atom_positions: np.ndarray  # [n_atoms, 3]
    atom_names: np.ndarray  # [n_atoms, 1]
    atom_elements: np.ndarray  # [n_atoms, 1]
    atom_residues: np.ndarray  # [n_atoms, 1] if ligand: x
    residue_idxs: np.ndarray #[n_atoms, 1]
    chain_ids: np.ndarray #[n_atoms, 1]
    is_ligand: np.ndarray  # [n_atoms, 1]
    metal_positions: np.ndarray  # [n_metals, 3]
    metal_types: np.ndarray  # [n_metals, 1]
    grid_positions: np.ndarray #[n_grids, 3]
    
def read_pdb(pdb_path) -> Structure:
    with open(pdb_path, "r") as f:
        pdb_str = f.read()
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("none", pdb_fh)
    model = list(structure.get_models())[0]

    data = defaultdict(list)

    for chain in model:
        for res in chain:
            if res.id[2] != " ":
                raise ValueError(f"Insertion code found at chain {chain.id}, residue {res.id[1]}")
            if res.id[0] == " ":  # ATOM
                for atom in res:
                    data["atom_positions"].append(atom.coord)
                    data["atom_elements"].append(atom.element)
                    data["atom_residues"].append(res.get_resname())
                    data["atom_names"].append(atom.name)
                    data["is_ligand"].append(0)
                    data["residue_idxs"].append(res.get_id()[1])
                    data["chain_ids"].append(chain.get_id())
            elif "H_" in res.id[0]:  # HETATM except water (which starts with "W_")
                for atom in res.get_atoms():
                    if atom.element in metals and atom.element in res.get_resname():
                        data["metal_positions"].append(atom.coord)
                        data["metal_types"].append(atom.element)
                    else:  # Ligand
                        data["atom_positions"].append(atom.coord)
                        data["atom_elements"].append(atom.element)
                        data["atom_residues"].append(res.get_resname())
                        data["atom_names"].append(atom.name)
                        data["is_ligand"].append(1)
                        data["residue_idxs"].append(res.get_id()[1])
                        data["chain_ids"].append(chain.get_id())

    return Structure(**{k: np.array(v) for k, v in data.items()})