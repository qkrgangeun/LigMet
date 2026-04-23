#!/usr/bin/env python3
"""
MMseqs2 clustering + metal local-site table builder.

This script does only these steps:
1) MMseqs2 clustering on training FASTA
2) Save cluster assignment table
3) Build per-metal annotation table with:
   - cluster id
   - metal coordinates/types
   - coordinating residues within 3A
   - local-site residues within 5A
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Local import
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from pdb_structure_utils import get_local_site_residues
except ImportError as exc:
    raise ImportError(f"Failed to import pdb_structure_utils: {exc}") from exc


logger = logging.getLogger(__name__)


def setup_logging(*, verbose: bool) -> None:
    """Configure logging.

    Args:
        verbose: Whether to enable debug-level logs.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run MMseqs2 clustering and export clustered metal local-site annotations."
    )
    parser.add_argument(
        "--train-pdb-list",
        type=Path,
        default=Path("/home/qkrgangeun/LigMet/code/text/biolip/filtered/train_pdbs_chain_1_filtered.txt"),
        help="Training PDB list (one pdb id per line).",
    )
    parser.add_argument(
        "--fasta-input",
        type=Path,
        default=Path("/home/qkrgangeun/LigMet/data/biolip_backup/fasta/train_chain1/fasta_input.txt"),
        help="Full FASTA input file.",
    )
    parser.add_argument(
        "--metal-label-dir",
        type=Path,
        default=Path("/home/qkrgangeun/LigMet/data/biolip_backup/metal_label"),
        help="Directory containing per-PDB metal label npz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/qkrgangeun/LigMet/revision/mmseqs2_analysis"),
        help="Output directory.",
    )
    parser.add_argument("--identity", type=float, default=0.95, help="MMseqs2 minimum sequence identity.")
    parser.add_argument("--coverage", type=float, default=0.9, help="MMseqs2 coverage threshold.")
    parser.add_argument("--threads", type=int, default=4, help="MMseqs2 thread count.")
    parser.add_argument("--coord-radius", type=float, default=3.0, help="Coordinating residue radius (A).")
    parser.add_argument("--site-radius", type=float, default=5.0, help="Local-site residue radius (A).")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def load_training_pdbs(*, train_pdb_list: Path) -> list[str]:
    """Load training PDB IDs.

    Args:
        train_pdb_list: Path to training PDB list file.

    Returns:
        Lower-cased PDB IDs.
    """
    with train_pdb_list.open(mode="r", encoding="utf-8") as handle:
        pdb_ids = [line.strip().lower() for line in handle if line.strip()]
    logger.info("Loaded %d training PDB IDs", len(pdb_ids))
    return pdb_ids


def extract_pdb_id_from_header(*, header_text: str) -> str:
    """Extract PDB ID from FASTA/MMseq header text.

    Supported examples:
    - 5fsp:A
    - 5fsp_A
    - 5fsp

    Args:
        header_text: Header content without leading '>'.

    Returns:
        Lower-cased PDB ID token.
    """
    token = header_text.strip()
    if not token:
        return ""

    if ":" in token:
        token = token.split(":", maxsplit=1)[0]
    if "_" in token:
        token = token.split("_", maxsplit=1)[0]
    return token.lower()


def prepare_fasta_for_mmseqs2(*, train_pdbs: list[str], fasta_input: Path, output_dir: Path) -> Path:
    """Filter full FASTA to training PDBs only.

    Args:
        train_pdbs: PDB IDs in training set.
        fasta_input: Source FASTA path.
        output_dir: Output directory.

    Returns:
        Filtered FASTA path.
    """
    pdb_set = set(train_pdbs)
    filtered_fasta = output_dir / "train_filtered.fasta"

    kept_headers = 0
    total_headers = 0
    current_pdb = ""
    keep_seq = False

    with fasta_input.open(mode="r", encoding="utf-8") as infile:
        with filtered_fasta.open(mode="w", encoding="utf-8") as outfile:
            for raw_line in infile:
                line = raw_line.rstrip("\n")
                if line.startswith(">"):
                    total_headers += 1
                    current_pdb = extract_pdb_id_from_header(header_text=line[1:])
                    keep_seq = current_pdb in pdb_set
                    if keep_seq:
                        outfile.write(line + "\n")
                        kept_headers += 1
                elif keep_seq:
                    outfile.write(line + "\n")

    if kept_headers == 0:
        raise ValueError(
            "Filtered FASTA has 0 entries. Check PDB list/header format overlap. "
            f"total_headers={total_headers}, matched_headers={kept_headers}"
        )

    logger.info(
        "Filtered FASTA saved: %s (headers kept=%d / total=%d)",
        filtered_fasta,
        kept_headers,
        total_headers,
    )
    return filtered_fasta


def run_mmseqs2_clustering(
    *,
    fasta_file: Path,
    output_dir: Path,
    identity: float,
    coverage: float,
    threads: int,
) -> tuple[Path, Path, Path]:
    """Run MMseqs2 createdb/cluster/createtsv.

    Args:
        fasta_file: Filtered FASTA file.
        output_dir: Output directory.
        identity: Minimum sequence identity.
        coverage: Coverage threshold.
        threads: Thread count.

    Returns:
        Paths for db, cluster db, and cluster TSV.
    """
    db = output_dir / "mmseqs_db"
    clusters = output_dir / "mmseqs_clusters"
    tmp_dir = output_dir / "tmp"
    cluster_tsv = output_dir / "mmseqs_clusters.tsv"

    subprocess.run([
        "mmseqs",
        "createdb",
        str(fasta_file),
        str(db),
    ], check=True)

    subprocess.run([
        "mmseqs",
        "cluster",
        str(db),
        str(clusters),
        str(tmp_dir),
        "--min-seq-id",
        str(identity),
        "-c",
        str(coverage),
        "--threads",
        str(threads),
    ], check=True)

    subprocess.run([
        "mmseqs",
        "createtsv",
        str(db),
        str(db),
        str(clusters),
        str(cluster_tsv),
    ], check=True)

    logger.info("MMseqs2 clustering done: %s", cluster_tsv)
    return db, clusters, cluster_tsv


def extract_pdb_id_from_seq_id(*, seq_id: str) -> str:
    """Extract PDB ID from MMseq sequence ID string.

    Args:
        seq_id: Sequence id from FASTA header/MMseq output.

    Returns:
        Lower-cased PDB ID.
    """
    cleaned = seq_id.strip()
    if cleaned.startswith(">"):
        cleaned = cleaned[1:]
    return extract_pdb_id_from_header(header_text=cleaned)


def parse_mmseqs_cluster_assignments(*, cluster_tsv: Path) -> pd.DataFrame:
    """Parse MMseqs cluster TSV into assignment table.

    Args:
        cluster_tsv: MMseqs createtsv output path.

    Returns:
        Assignment dataframe.
    """
    rows: list[dict[str, str]] = []
    with cluster_tsv.open(mode="r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            rep_id = parts[0]
            member_id = parts[1]
            rows.append(
                {
                    "cluster95_id": rep_id,
                    "member_seq_id": member_id,
                    "pdb_id": extract_pdb_id_from_seq_id(seq_id=member_id),
                }
            )

    df = pd.DataFrame(rows).drop_duplicates(ignore_index=True)
    logger.info("Parsed cluster assignments: rows=%d, clusters=%d", len(df), df["cluster95_id"].nunique())
    return df


def load_metal_annotation(*, pdb_id: str, metal_label_dir: Path) -> dict[str, np.ndarray] | None:
    """Load metal annotation NPZ for one PDB.

    Args:
        pdb_id: PDB identifier.
        metal_label_dir: Directory containing NPZ files.

    Returns:
        Dictionary with positions/types arrays or None.
    """
    npz_path = metal_label_dir / f"{pdb_id.lower()}.npz"
    if not npz_path.exists():
        return None

    try:
        data = np.load(npz_path, allow_pickle=True)
        positions = np.asarray(data["metal_positions"]) if "metal_positions" in data else np.empty((0, 3))
        raw_types = data["metal_types"] if "metal_types" in data else np.asarray([])
        metal_types = np.asarray(raw_types).astype(str)
        data.close()
    except Exception as exc:
        logger.warning("Failed to load metal annotation for %s: %s", pdb_id, exc)
        return None

    return {
        "positions": positions,
        "types": metal_types,
    }


def residues_to_string(*, residues: set[tuple[str, int]]) -> str:
    """Serialize residue set to stable string.

    Args:
        residues: Set of (chain_id, residue_number).

    Returns:
        Serialized residue signature string.
    """
    sorted_residues = sorted(list(residues), key=lambda item: (item[0], item[1]))
    return ";".join([f"{chain}:{resnum}" for chain, resnum in sorted_residues])


def build_clustered_site_table(
    *,
    cluster_df: pd.DataFrame,
    metal_label_dir: Path,
    coord_radius: float,
    site_radius: float,
) -> pd.DataFrame:
    """Build per-metal site table with cluster id and local residues.

    Args:
        cluster_df: MMseq assignment dataframe.
        metal_label_dir: Directory with metal annotation NPZ files.
        coord_radius: Radius for coordinating residues.
        site_radius: Radius for local site residues.

    Returns:
        Site-level dataframe.
    """
    cluster_group = cluster_df.groupby("pdb_id")["cluster95_id"].unique()
    pdb_to_cluster = {pdb_id: sorted(list(cluster_ids))[0] for pdb_id, cluster_ids in cluster_group.items()}

    rows: list[dict[str, object]] = []
    entry_id = 0

    for pdb_id, cluster_id in sorted(pdb_to_cluster.items(), key=lambda item: (item[1], item[0])):
        annotation = load_metal_annotation(pdb_id=pdb_id, metal_label_dir=metal_label_dir)
        if annotation is None:
            continue

        positions = annotation["positions"]
        types = annotation["types"]
        if len(positions) == 0:
            continue

        for metal_idx in range(len(positions)):
            metal_pos = positions[metal_idx]
            metal_type = str(types[metal_idx]) if metal_idx < len(types) else "NA"

            coord_dict = get_local_site_residues(
                pdb_id=pdb_id,
                metal_positions=np.asarray([metal_pos]),
                radius=coord_radius,
            )
            site_dict = get_local_site_residues(
                pdb_id=pdb_id,
                metal_positions=np.asarray([metal_pos]),
                radius=site_radius,
            )

            coord_residues = coord_dict.get(0, set())
            site_residues = site_dict.get(0, set())

            rows.append(
                {
                    "entry_id": entry_id,
                    "pdb_id": pdb_id,
                    "cluster95_id": cluster_id,
                    "metal_idx": metal_idx,
                    "metal_type": metal_type,
                    "metal_x": float(metal_pos[0]),
                    "metal_y": float(metal_pos[1]),
                    "metal_z": float(metal_pos[2]),
                    "coord_residues_3A": residues_to_string(residues=coord_residues),
                    "site_residues_5A": residues_to_string(residues=site_residues),
                    "n_coord_residues_3A": len(coord_residues),
                    "n_site_residues_5A": len(site_residues),
                }
            )
            entry_id += 1

    site_df = pd.DataFrame(rows)
    logger.info("Built site table: rows=%d", len(site_df))
    return site_df


def main() -> None:
    """Run clustering + per-site CSV generation only."""
    args = parse_args()
    setup_logging(verbose=args.verbose)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_pdbs = load_training_pdbs(train_pdb_list=args.train_pdb_list)
    filtered_fasta = prepare_fasta_for_mmseqs2(
        train_pdbs=train_pdbs,
        fasta_input=args.fasta_input,
        output_dir=args.output_dir,
    )

    _, _, cluster_tsv = run_mmseqs2_clustering(
        fasta_file=filtered_fasta,
        output_dir=args.output_dir,
        identity=args.identity,
        coverage=args.coverage,
        threads=args.threads,
    )

    cluster_df = parse_mmseqs_cluster_assignments(cluster_tsv=cluster_tsv)
    cluster_csv = args.output_dir / "mmseqs_cluster_assignments.csv"
    cluster_df.to_csv(cluster_csv, index=False)
    logger.info("Saved cluster assignment CSV: %s", cluster_csv)

    site_df = build_clustered_site_table(
        cluster_df=cluster_df,
        metal_label_dir=args.metal_label_dir,
        coord_radius=args.coord_radius,
        site_radius=args.site_radius,
    )
    site_csv = args.output_dir / "training_metal_sites_with_clusters.csv"
    site_df.to_csv(site_csv, index=False)
    logger.info("Saved clustered metal-site CSV: %s", site_csv)


if __name__ == "__main__":
    main()
