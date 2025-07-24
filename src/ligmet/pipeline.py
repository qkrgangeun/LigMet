# src/ligmet/pipeline.py

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import asdict
from joblib import load

from ligmet.utils.pdb import read_pdb, StructureWithGrid  # type: ignore
from ligmet.featurizer import make_features, Features  # type: ignore
from ligmet.utils.grid import sasa_grids_thread, filter_by_clashmap  # type: ignore
from ligmet.utils.io import load_npz  # type: ignore
from ligmet.utils.rf.rf_features import (
    near_lig,
    near_res,
    nearest_bb_dist,
    nearest_prot_carbon_dist,
    binned_res,
    parse_pdb,
    filter_by_biometall,
    RSA,
)  # type: ignore
from ligmet.utils.constants import aliphatic_carbons, aromatic_carbons
from ligmet.utils.rf.label import label_grids  # type: ignore

def extract_dl_features(pdb_path: Path) -> dict[str, np.ndarray]:
    """
    Read a PDB, build grid & structure, compute DL features,
    and return a dict of {feature_name: ndarray}.
    """
    struct = read_pdb(pdb_path)
    if len(struct.atom_positions) > 50000:
        raise RuntimeError(f"Too many atoms in {pdb_path.name} (>50000)")

    # build & filter grids
    grids = sasa_grids_thread(struct.atom_positions, struct.atom_elements)
    grids = filter_by_clashmap(grids)

    # ensure metal fields exist
    sd = asdict(struct)
    sd.setdefault("metal_positions", np.empty((0, 3)))
    sd.setdefault("metal_types",    np.empty((0,), dtype=object))

    # wrap into StructureWithGrid
    struct_wg = StructureWithGrid(grid_positions=grids, **sd)
    feats = make_features(pdb_path, struct_wg)

    # extract only ndarray fields
    return {k: v for k, v in asdict(feats).items() if isinstance(v, np.ndarray)}


def extract_rf_features(npz_path: str, cutoff: float = 2.0) -> pd.DataFrame:
    """
    Load a .npz of DL features, compute RF features and labels,
    return as a pandas DataFrame.
    """
    raw = load_npz(npz_path)
    struct = Features(**raw)

    protein_mask = struct.is_ligand == 0
    ligand_mask  = struct.is_ligand == 1
    df = pd.DataFrame()

    # spatial residue+ligand features
    for t in [2.5, 2.8, 3.0, 3.2, 5.0]:
        p_coords, p_core, p_bb = near_res(
            struct.atom_residues[protein_mask],
            struct.atom_names[protein_mask],
            struct.atom_elements[protein_mask],
            struct.atom_positions[protein_mask],
            struct.grid_positions,
            t
        )
        n_nos, n_nion, n_etc = near_lig(
            struct.atom_positions[ligand_mask],
            struct.atom_elements[ligand_mask],
            struct.atom_residues[ligand_mask],
            struct.grid_positions,
            t
        )
        df[f"p_coords_res_{t}"] = np.array(p_coords, dtype=np.int8)
        df[f"p_core_res_{t}"]   = np.array(p_core, dtype=np.int8)
        df[f"p_bb_coords_{t}"]  = np.array(p_bb, dtype=np.int8)
        df[f"n_lig_NOS_{t}"]     = np.array(n_nos, dtype=np.int8)
        df[f"n_lig_nion_{t}"]    = np.array(n_nion, dtype=np.int8)
        df[f"n_lig_etc_{t}"]     = np.array(n_etc, dtype=np.int8)

    # binned residue counts
    coords, core, bb = binned_res(
        struct.atom_residues,
        struct.atom_names,
        struct.atom_elements,
        struct.atom_positions,
        struct.grid_positions,
        3,
        5
    )
    df["n_coords_res_bin"] = np.array(coords, dtype=np.int8)
    df["n_core_res_bin"]   = np.array(core,  dtype=np.int8)
    df["n_bb_coords_bin"]  = np.array(bb,    dtype=np.int8)

    # distance metrics
    df["min_c_dist"] = np.array(
        nearest_prot_carbon_dist(
            struct.atom_residues,
            struct.atom_names,
            struct.atom_elements,
            struct.atom_positions,
            struct.grid_positions,
            aliphatic_carbons,
            aromatic_carbons
        ),
        dtype=np.float16
    )
    df["near_bb_dist"] = np.array(
        nearest_bb_dist(
            struct.atom_names[protein_mask],
            struct.atom_positions[protein_mask],
            struct.grid_positions
        ),
        dtype=np.float16
    )

    # SASA and biometall
    df["sasa"] = np.array(
        RSA(struct.grid_positions, struct.atom_positions, struct.atom_elements),
        dtype=np.float16
    )
    atom_dict, grids = parse_pdb(struct)
    df["biometall"] = np.array(
        filter_by_biometall(grids, atom_dict),
        dtype=np.int8
    )

    # label grids safely
    df["label_2.0"] = label_grids(
        struct.metal_positions,
        struct.grid_positions,
        t=cutoff
    )

    return df


def run_rf_inference(csv_path:str, model_path: str) -> np.ndarray:
    """
    Load RF model, read csv.gz features for pdb_id in feat_dir,
    run predict_proba and return probability array.
    """
    model_path = model_path 
    model = load(model_path)

    df = pd.read_csv(csv_path, compression="gzip")
    X = df.drop(columns=["label_2.0"])
    probs = model.predict_proba(X)[:, 1]
    return probs
