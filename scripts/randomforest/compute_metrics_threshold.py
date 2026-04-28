#!/usr/bin/env python3
"""Compute threshold metrics for RandomForest and LightGBM.

Metrics per threshold:
1) metal_site_coverage: covered_metal_sites / total_metal_sites
2) grid_filtering_ratio: grids_after_filter / total_grids
3) false_true_ratio: false_grids_after_filter / true_grids_after_filter
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from joblib import load
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm


@dataclass
class RunningCounts:
    total_metal_sites: int
    covered_metal_sites: dict[float, int]
    total_grids: int
    grids_after_filter: dict[float, int]
    false_grids_after_filter: dict[float, int]
    true_grids_after_filter: dict[float, int]


def setup_logging(*, verbose: bool) -> None:
    """Configure logging level.

    Args:
        verbose: Enable debug logs when True.
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="[%(levelname)s] %(message)s")


def read_pdb_list(*, list_path: Path) -> list[str]:
    """Read PDB IDs from a text file.

    Args:
        list_path: Path to a text file with one PDB ID per line.

    Returns:
        Parsed PDB ID list.
    """
    with list_path.open(mode="r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def init_counts(*, thresholds: list[float]) -> RunningCounts:
    """Initialize metric accumulators.

    Args:
        thresholds: Probability thresholds.

    Returns:
        Empty running counts.
    """
    return RunningCounts(
        total_metal_sites=0,
        covered_metal_sites={threshold: 0 for threshold in thresholds},
        total_grids=0,
        grids_after_filter={threshold: 0 for threshold in thresholds},
        false_grids_after_filter={threshold: 0 for threshold in thresholds},
        true_grids_after_filter={threshold: 0 for threshold in thresholds},
    )


def build_model_probability(
    *,
    model,
    feature_frame: pd.DataFrame,
    label_column: str,
) -> np.ndarray:
    """Predict LightGBM probabilities for one PDB.

    Args:
        model: Trained LightGBM model.
        feature_frame: Feature dataframe including label column.
        label_column: Label column to drop before inference.

    Returns:
        Probability array for positive class.
    """
    features = feature_frame.drop(columns=[label_column])
    return model.predict_proba(features)[:, 1]


def build_lgbm_probability(
    *,
    model,
    feature_frame: pd.DataFrame,
    label_column: str,
) -> np.ndarray:
    """Backward-compatible wrapper for LightGBM-like models."""
    return build_model_probability(model=model, feature_frame=feature_frame, label_column=label_column)


def load_rf_probability(*, rf_prob_dir: Path, pdb_id: str) -> np.ndarray | None:
    """Load precomputed RandomForest probability for one PDB.

    Args:
        rf_prob_dir: Directory containing {pdb_id}.npz with key 'prob'.
        pdb_id: Target PDB ID.

    Returns:
        Probability array if available, else None.
    """
    prob_path = rf_prob_dir / f"{pdb_id}.npz"
    if not prob_path.exists():
        return None
    data = np.load(prob_path)
    if "prob" not in data:
        return None
    return np.asarray(data["prob"])


def apply_threshold_metrics(
    *,
    counts: RunningCounts,
    thresholds: list[float],
    probabilities: np.ndarray,
    labels: np.ndarray,
    grid_positions: np.ndarray,
    metal_positions: np.ndarray,
) -> None:
    """Accumulate threshold metrics for one PDB.

    Args:
        counts: Global counters updated in-place.
        thresholds: Threshold values.
        probabilities: Per-grid probabilities.
        labels: Per-grid binary labels.
        grid_positions: Grid coordinates for this PDB.
        metal_positions: Metal coordinates for this PDB.
    """
    counts.total_metal_sites += int(metal_positions.shape[0])
    counts.total_grids += int(probabilities.shape[0])

    for threshold in thresholds:
        pred_mask = probabilities >= threshold
        kept_count = int(np.sum(pred_mask))
        counts.grids_after_filter[threshold] += kept_count

        kept_labels = labels[pred_mask]
        counts.false_grids_after_filter[threshold] += int(np.sum(kept_labels == 0))
        counts.true_grids_after_filter[threshold] += int(np.sum(kept_labels == 1))

        if kept_count == 0 or metal_positions.size == 0:
            continue

        pred_positions = grid_positions[pred_mask]
        tree = cKDTree(pred_positions)
        covered = 0
        for metal_xyz in metal_positions:
            nearby = tree.query_ball_point(metal_xyz, r=2.0)
            if nearby:
                covered += 1
        counts.covered_metal_sites[threshold] += covered


def finalize_result_rows(*, model_name: str, thresholds: list[float], counts: RunningCounts) -> list[dict[str, object]]:
    """Convert counters into result rows.

    Args:
        model_name: Model name label.
        thresholds: Threshold values.
        counts: Accumulated counters.

    Returns:
        List of output rows.
    """
    rows: list[dict[str, object]] = []
    total_metal_sites = max(counts.total_metal_sites, 1)
    total_grids = max(counts.total_grids, 1)

    for threshold in thresholds:
        covered = counts.covered_metal_sites[threshold]
        kept = counts.grids_after_filter[threshold]
        false_count = counts.false_grids_after_filter[threshold]
        true_count = counts.true_grids_after_filter[threshold]

        row = {
            "model": model_name,
            "threshold": float(threshold),
            "metal_site_coverage": float(covered / total_metal_sites),
            "covered_metal_sites": int(covered),
            "total_metal_sites": int(counts.total_metal_sites),
            "grids_after_filtering": int(kept),
            "total_grids": int(counts.total_grids),
            "grid_filtering_ratio": float(kept / total_grids),
            "false_grids": int(false_count),
            "true_grids": int(true_count),
            "false_true_ratio": float(false_count / max(true_count, 1)),
        }
        rows.append(row)

    return rows


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Compute threshold metrics quickly with per-PDB streaming.")
    parser.add_argument(
        "--rf_prob_dir",
        type=Path,
        default=Path("/home/qkrgangeun/LigMet/data/biolip_backup/rf/grid_prob"),
    )
    parser.add_argument(
        "--lgbm_model_path",
        type=Path,
        default=Path("/home/qkrgangeun/LigMet/data/rf_param/lightgbm_revision.joblib"),
    )
    parser.add_argument(
        "--mlp_model_path",
        type=Path,
        default=Path("/home/qkrgangeun/LigMet/data/rf_param/mlp_baseline.joblib"),
    )
    parser.add_argument(
        "--test_pdb_list",
        type=Path,
        default=Path("/home/qkrgangeun/LigMet/data/biolip_backup/pdb/test_pdb_noerror.txt"),
    )
    parser.add_argument(
        "--test_feature_dir",
        type=Path,
        default=Path("/home/qkrgangeun/LigMet/data/biolip_backup/rf/features"),
    )
    parser.add_argument(
        "--dl_feature_dir",
        type=Path,
        default=Path("/home/qkrgangeun/LigMet/data/biolip_backup/dl/features"),
    )
    parser.add_argument(
        "--metal_label_dir",
        type=Path,
        default=Path("/home/qkrgangeun/LigMet/data/biolip_backup/metal_label"),
    )
    parser.add_argument("--label_column", type=str, default="label_2.0")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=["rf", "lgbm", "mlp"],
        default=["rf", "lgbm", "mlp"],
        help="Select which models to evaluate. Use only 'mlp' to test the MLP joblib.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("/home/qkrgangeun/LigMet/data/rf_param/metrics/threshold_comparison_metrics2.csv"),
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run threshold metric computation."""
    args = parse_args()
    setup_logging(verbose=args.verbose)

    thresholds = sorted(args.thresholds)
    pdb_ids = read_pdb_list(list_path=args.test_pdb_list)
    selected_models = set(args.models)

    lgbm_model = load(args.lgbm_model_path) if "lgbm" in selected_models else None
    mlp_model = load(args.mlp_model_path) if "mlp" in selected_models else None

    rf_counts = init_counts(thresholds=thresholds) if "rf" in selected_models else None
    lgbm_counts = init_counts(thresholds=thresholds) if "lgbm" in selected_models else None
    mlp_counts = init_counts(thresholds=thresholds) if "mlp" in selected_models else None

    missing_rf_prob_count = 0
    skipped_pdb_count = 0

    for pdb_id in tqdm(pdb_ids, desc="Processing PDBs"):
        feature_path = args.test_feature_dir / f"{pdb_id}.csv.gz"
        dl_npz_path = args.dl_feature_dir / f"{pdb_id}.npz"
        metal_label_npz_path = args.metal_label_dir / f"{pdb_id}.npz"
        if not feature_path.exists() or not dl_npz_path.exists() or not metal_label_npz_path.exists():
            skipped_pdb_count += 1
            continue

        frame = pd.read_csv(feature_path, compression="gzip")
        if args.label_column not in frame.columns:
            logging.warning("Missing label column in %s", feature_path)
            skipped_pdb_count += 1
            continue

        labels = frame[args.label_column].to_numpy(dtype=int)
        rf_prob = None
        lgbm_prob = None
        mlp_prob = None

        if "rf" in selected_models:
            rf_prob = load_rf_probability(
                rf_prob_dir=args.rf_prob_dir,
                pdb_id=pdb_id,
            )
            if rf_prob is None:
                missing_rf_prob_count += 1
                continue

        if "lgbm" in selected_models:
            assert lgbm_model is not None
            lgbm_prob = build_lgbm_probability(
                model=lgbm_model,
                feature_frame=frame,
                label_column=args.label_column,
            )

        if "mlp" in selected_models:
            assert mlp_model is not None
            mlp_prob = build_model_probability(
                model=mlp_model,
                feature_frame=frame,
                label_column=args.label_column,
            )

        dl_data = np.load(dl_npz_path)
        metal_data = np.load(metal_label_npz_path)
        if "grid_positions" not in dl_data or "metal_positions" not in metal_data:
            skipped_pdb_count += 1
            continue

        grid_positions = np.asarray(dl_data["grid_positions"])
        metal_positions = np.asarray(metal_data["metal_positions"])

        lengths = [len(labels), len(grid_positions)]
        if rf_prob is not None:
            lengths.append(len(rf_prob))
        if lgbm_prob is not None:
            lengths.append(len(lgbm_prob))
        if mlp_prob is not None:
            lengths.append(len(mlp_prob))

        n = min(lengths)
        if n == 0:
            skipped_pdb_count += 1
            continue

        mismatch_parts = [f"labels={len(labels)}", f"grid={len(grid_positions)}"]
        if rf_prob is not None:
            mismatch_parts.append(f"rf={len(rf_prob)}")
        if lgbm_prob is not None:
            mismatch_parts.append(f"lgbm={len(lgbm_prob)}")
        if mlp_prob is not None:
            mismatch_parts.append(f"mlp={len(mlp_prob)}")

        if any(length != n for length in lengths):
            logging.debug("Length mismatch for %s: %s, using n=%d", pdb_id, " ".join(mismatch_parts), n)

        labels_n = labels[:n]
        grid_positions_n = grid_positions[:n]
        if rf_prob is not None and rf_counts is not None:
            apply_threshold_metrics(
                counts=rf_counts,
                thresholds=thresholds,
                probabilities=rf_prob[:n],
                labels=labels_n,
                grid_positions=grid_positions_n,
                metal_positions=metal_positions,
            )
        if lgbm_prob is not None and lgbm_counts is not None:
            apply_threshold_metrics(
                counts=lgbm_counts,
                thresholds=thresholds,
                probabilities=lgbm_prob[:n],
                labels=labels_n,
                grid_positions=grid_positions_n,
                metal_positions=metal_positions,
            )
        if mlp_prob is not None and mlp_counts is not None:
            apply_threshold_metrics(
                counts=mlp_counts,
                thresholds=thresholds,
                probabilities=mlp_prob[:n],
                labels=labels_n,
                grid_positions=grid_positions_n,
                metal_positions=metal_positions,
            )

    rows = []
    if rf_counts is not None:
        rows.extend(finalize_result_rows(model_name="RandomForest", thresholds=thresholds, counts=rf_counts))
    if lgbm_counts is not None:
        rows.extend(finalize_result_rows(model_name="LightGBM", thresholds=thresholds, counts=lgbm_counts))
    if mlp_counts is not None:
        rows.extend(finalize_result_rows(model_name="MLP", thresholds=thresholds, counts=mlp_counts))

    result_frame = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(args.output_csv, index=False)

    logging.info("Saved results to %s", args.output_csv)
    logging.info("Skipped PDB count: %d", skipped_pdb_count)
    logging.info("Missing RF probability count: %d", missing_rf_prob_count)

    print("\n=== RESULTS ===\n")
    print(result_frame.to_string(index=False))


if __name__ == "__main__":
    main()
