import numpy as np
from pathlib import Path
from ligmet.utils.pdb import read_pdb, Structure, StructureWithGrid
from ligmet.utils.grid import *
from ligmet.featurizer import * # type: ignore
from openbabel import openbabel
from dataclasses import asdict
import traceback
import argparse

# 경로 설정
pdb_dir = Path("/home/qkrgangeun/LigMet/data/biolip/pdb")
npz_dir = Path("/home/qkrgangeun/LigMet/data/biolip/dl/features")

# Features 클래스 기준 필드 목록
ALL_FEATURE_KEYS = [
    "atom_positions", "atom_names", "atom_elements", "atom_residues",
    "residue_idxs", "chain_ids", "is_ligand", "metal_positions",
    "metal_types", "grid_positions", "sasas", "qs",
    "sec_structs", "gen_types", "bond_masks"
]

def bondmask_to_neighidx(bond_mask: np.ndarray) -> np.ndarray:
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
    elif key in ["atom_elements", "atom_residues"]:
        return arr.astype("<U3")
    elif key == "atom_names":
        return arr.astype("<U4")
    elif key == "chain_ids":
        return arr.astype("<U1")
    elif arr.dtype.kind == "U":
        maxlen = max(len(str(s)) for s in arr)
        return arr.astype(f"<U{maxlen}")
    return arr

def make_structure_bondmask(pdb_path: Path, structure: Structure):
    pdb_io, protein_io, ligand_io = make_pdb(structure)
    ligand_pdb_str = ligand_io.getvalue()

    ligand_mol = None
    if ligand_pdb_str.strip():
        ob_conversion = openbabel.OBConversion()
        ob_conversion.SetInFormat("pdb")
        ob_mol = openbabel.OBMol()
        ob_conversion.ReadString(ob_mol, ligand_pdb_str)
        ligand_mol = ob_mol

    new_pdb_path = process_pdb(pdb_io)
    new_structure = read_pdb(new_pdb_path)
    bond_masks = cov_bonds_mask(new_structure, ligand_mol)
    return bond_masks, new_structure

def fix_features_preserve_all(pdb_id: str):
    pdb_path = pdb_dir / f"{pdb_id}.pdb"
    npz_path = npz_dir / f"{pdb_id}.npz"

    if not npz_path.exists():
        print(f"❌ {npz_path} does not exist. Rebuilding from scratch...")
        return process_original_pdb(pdb_id)

    try:
        print(f"🔧 Fixing features for: {pdb_id}")

        old = dict(np.load(npz_path))
        missing_keys = [k for k in ALL_FEATURE_KEYS if k not in old]
        if missing_keys:
            print(f"❌ Missing keys: {missing_keys}, regenerating...")
            return process_original_pdb(pdb_id)

        structure = read_pdb(pdb_path)
        bond_masks, structure_new = make_structure_bondmask(pdb_path, structure)

        updated_fields = {
            "atom_positions": np.array(structure_new.atom_positions, dtype=np.float32),
            "atom_names": np.array(structure_new.atom_names, dtype="<U4"),
            "atom_elements": np.array(structure_new.atom_elements, dtype="<U3"),
            "atom_residues": np.array(structure_new.atom_residues, dtype="<U3"),
            "residue_idxs": np.array(structure_new.residue_idxs, dtype=np.int32),
            "chain_ids": np.array(structure_new.chain_ids, dtype="<U1"),
            "is_ligand": np.array(structure_new.is_ligand, dtype=np.bool_),
            "bond_masks": bond_masks,
        }

        final_dict = {k: optimize_dtype(k, updated_fields.get(k, old[k])) for k in ALL_FEATURE_KEYS}

        np.savez(npz_path, **final_dict)
        print(f"✅ Fixed and saved: {npz_path}")

    except Exception as e:
        print(f"⚠️ Error during fix. Falling back to full regeneration for {pdb_id}")
        traceback.print_exc()
        process_original_pdb(pdb_id)

def process_original_pdb(pdb_id: str):
    pdb_path = pdb_dir / f"{pdb_id}.pdb"
    npz_path = npz_dir / f"{pdb_id}.npz"

    try:
        print(f"📂 Processing from scratch: {pdb_id}")
        structure = read_pdb(pdb_path)

        if len(structure.atom_positions) > 50000:
            print(f"⚠️ Skipping {pdb_id}, too many atoms")
            return

        grids = sasa_grids_thread(structure.atom_positions, structure.atom_elements)
        grids = filter_by_clashmap(grids)

        structure_with_grid = StructureWithGrid(
            grid_positions=grids,
            **asdict(structure)
        )

        features = make_features(pdb_path, structure_with_grid)

        feature_dict = {
            k: optimize_dtype(k, v)
            for k, v in asdict(features).items()
            if isinstance(v, np.ndarray)
        }

        np.savez(npz_path, **feature_dict)
        print(f"✅ Rebuilt and saved: {npz_path}")

    except Exception as e:
        print(f"❌ Failed to process {pdb_id}: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdb_id", type=str)
    args = parser.parse_args()
    fix_features_preserve_all(args.pdb_id)
