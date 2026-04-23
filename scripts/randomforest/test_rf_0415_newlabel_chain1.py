#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from joblib import load
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm


@dataclass
class DatasetBundle:
    features: pd.DataFrame
    labels: pd.Series


def setup_logging(*, verbose: bool) -> None:
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="[%(levelname)s] %(message)s")


def read_pdb_list(*, list_path: Path) -> list[str]:
    with list_path.open(mode="r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def load_feature_frames(
    *,
    pdb_ids: list[str],
    feature_dir: Path,
    label_column: str,
    file_prefix: str,
) -> DatasetBundle:
    x_frames: list[pd.DataFrame] = []
    y_frames: list[pd.Series] = []

    for pdb_id in tqdm(pdb_ids):
        feature_path = feature_dir / f"{file_prefix}{pdb_id}.csv.gz"
        if not feature_path.exists():
            logging.warning("Feature file not found: %s", feature_path)
            continue

        frame = pd.read_csv(feature_path, compression="gzip")
        if label_column not in frame.columns:
            raise ValueError(f"Missing label column '{label_column}' in {feature_path}")

        x_frames.append(frame.drop(columns=[label_column]))
        y_frames.append(frame[label_column])

    if not x_frames:
        raise ValueError("No feature files were loaded. Check input list/path settings.")

    features = pd.concat(x_frames, axis=0, ignore_index=True)
    labels = pd.concat(y_frames, axis=0, ignore_index=True)
    return DatasetBundle(features=features, labels=labels)


def load_test_dataset(
    *,
    test_list_path: Path,
    test_feature_dir: Path,
    test_prefix: str,
    label_column: str,
) -> DatasetBundle:
    test_pdb_ids = read_pdb_list(list_path=test_list_path)
    return load_feature_frames(
        pdb_ids=test_pdb_ids,
        feature_dir=test_feature_dir,
        label_column=label_column,
        file_prefix=test_prefix,
    )


def evaluate_model(
    *,
    model: Any,
    test_bundle: DatasetBundle,
    threshold: float,
) -> dict[str, float]:
    if not hasattr(model, "predict_proba"):
        raise ValueError("Model does not support predict_proba; cannot evaluate with threshold.")

    y_true = test_bundle.labels.to_numpy(dtype=int)
    y_score = model.predict_proba(test_bundle.features)[:, 1]
    y_pred = (y_score >= threshold).astype(int)

    metrics = {
        "n_samples": float(len(y_true)),
        "positive_ratio": float(np.mean(y_true)),
        "accuracy": float(accuracy_score(y_true=y_true, y_pred=y_pred)),
        "precision": float(precision_score(y_true=y_true, y_pred=y_pred, zero_division=0)),
        "recall": float(recall_score(y_true=y_true, y_pred=y_pred, zero_division=0)),
        "f1": float(f1_score(y_true=y_true, y_pred=y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true=y_true, y_pred=y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true=y_true, y_pred=y_pred)),
        "auroc": float(roc_auc_score(y_true=y_true, y_score=y_score)),
        "auprc": float(average_precision_score(y_true=y_true, y_score=y_score)),
    }
    logging.info("Classification report:\n%s", classification_report(y_true, y_pred, zero_division=0))
    logging.info("Total grids: %d", len(y_pred))
    logging.info("True predicted grids: %d", int(np.sum(y_pred == 1)))
    logging.info("True ratio: %.4f", float(np.mean(y_pred)))
    return metrics


def save_metrics(*, metrics: dict[str, float], model_type: str, threshold: float, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    row = {"model_type": model_type, "threshold": threshold}
    row.update(metrics)
    pd.DataFrame([row]).to_csv(output_csv, index=False)
    logging.info("Saved metrics: %s", output_csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained RandomForest model on test data.")
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path("/home/qkrgangeun/LigMet/data/rf_param/0415_newlabel_chain1"),
        help="Trained model .joblib path",
    )
    parser.add_argument(
        "--test_data",
        type=Path,
        default=Path("/home/qkrgangeun/LigMet/data/biolip_backup/pdb/test_pdb_noerror.txt"),
        help="Text file containing PDB IDs, one per line",
    )
    parser.add_argument(
        "--test_feature_dir",
        type=Path,
        default=Path("/home/qkrgangeun/LigMet/data/biolip_backup/rf/features"),
        help="Directory containing test feature CSV.GZ files",
    )
    parser.add_argument(
        "--test_prefix",
        type=str,
        default="",
        help="Optional prefix for test feature filenames",
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default="label_2.0",
        help="Label column name to evaluate",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold",
    )
    parser.add_argument(
        "--metrics_out_csv",
        type=Path,
        default=Path("/home/qkrgangeun/LigMet/data/rf_param/metrics/test_rf_0415_newlabel_chain1_metrics.csv"),
        help="Output CSV path for metrics",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(verbose=args.verbose)

    logging.info("Loading model: %s", args.model_path)
    model = load(args.model_path)

    logging.info("Loading test data from: %s", args.test_data)
    test_bundle = load_test_dataset(
        test_list_path=args.test_data,
        test_feature_dir=args.test_feature_dir,
        test_prefix=args.test_prefix,
        label_column=args.label_column,
    )

    metrics = evaluate_model(
        model=model,
        test_bundle=test_bundle,
        threshold=args.threshold,
    )

    metric_text = ", ".join([f"{key}={value:.4f}" for key, value in metrics.items()])
    logging.info("Test evaluation: %s", metric_text)

    save_metrics(
        metrics=metrics,
        model_type="rf",
        threshold=args.threshold,
        output_csv=args.metrics_out_csv,
    )


if __name__ == "__main__":
    main()