#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from joblib import dump, load
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm


@dataclass
class DatasetBundle:
    features: pd.DataFrame
    labels: pd.Series


def setup_logging(*, verbose: bool) -> None:
    """Configure logging.

    Args:
        verbose: Whether to enable debug logs.
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="[%(levelname)s] %(message)s")


def read_pdb_list(*, list_path: Path) -> list[str]:
    """Read PDB IDs from a text file.

    Args:
        list_path: Path to text file with one PDB ID per line.

    Returns:
        Parsed PDB ID list.
    """
    with list_path.open(mode="r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def load_feature_frames(
    *,
    pdb_ids: list[str],
    feature_dir: Path,
    label_column: str,
    file_prefix: str,
) -> DatasetBundle:
    """Load features/labels from compressed CSV files.

    Args:
        pdb_ids: PDB IDs to load.
        feature_dir: Directory containing {prefix}{pdb_id}.csv.gz.
        label_column: Label column name.
        file_prefix: Optional filename prefix.

    Returns:
        Concatenated features and labels.
    """
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


def load_train_dataset(
    *,
    train_list_path: Path,
    train_feature_dir: Path,
    label_column: str,
) -> DatasetBundle:
    """Load training dataset from Biolip train features.

    Args:
        train_list_path: Main train PDB list path.
        train_feature_dir: Main train feature directory.
        label_column: Label column name.

    Returns:
        Train dataset bundle.
    """
    train_pdb_ids = read_pdb_list(list_path=train_list_path)
    return load_feature_frames(
        pdb_ids=train_pdb_ids,
        feature_dir=train_feature_dir,
        label_column=label_column,
        file_prefix="",
    )


def load_test_dataset(
    *,
    test_list_path: Path,
    test_feature_dir: Path,
    test_prefix: str,
    label_column: str,
) -> DatasetBundle:
    """Load test dataset from feature CSV.GZ files.

    Args:
        test_list_path: Test PDB list path.
        test_feature_dir: Test feature directory.
        test_prefix: Prefix for test feature filenames.
        label_column: Label column name.

    Returns:
        Loaded test dataset bundle.
    """
    test_pdb_ids = read_pdb_list(list_path=test_list_path)
    return load_feature_frames(
        pdb_ids=test_pdb_ids,
        feature_dir=test_feature_dir,
        label_column=label_column,
        file_prefix=test_prefix,
    )


def build_model(*, model_type: str, random_state: int, svm_c: float, svm_kernel: str, svm_gamma: str) -> Any:
    """Build a model instance for training.

    Args:
        model_type: One of lightgbm or svm.
        random_state: Random seed.
        svm_c: SVM C value.
        svm_kernel: SVM kernel type.
        svm_gamma: SVM gamma value.

    Returns:
        Initialized model object.
    """
    if model_type == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError(
                "lightgbm is not installed. Install it in your active environment first."
            ) from exc

        return LGBMClassifier(
            objective="binary",
            class_weight="balanced",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=-1,
            random_state=random_state,
        )

    if model_type == "svm":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "svc",
                    SVC(
                        C=svm_c,
                        kernel=svm_kernel,
                        gamma=svm_gamma,
                        class_weight="balanced",
                        probability=True,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    raise ValueError(f"Unsupported model_type: {model_type}")


def fit_and_save_model(
    *,
    model: Any,
    train_bundle: DatasetBundle,
    model_out_path: Path,
) -> None:
    """Train model and save it to disk.

    Args:
        model: Model instance.
        train_bundle: Training dataset bundle.
        model_out_path: Path to save trained model.
    """
    logging.info("Train samples: %d, features: %d", len(train_bundle.labels), train_bundle.features.shape[1])
    model.fit(train_bundle.features, train_bundle.labels)
    model_out_path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, model_out_path)
    logging.info("Saved model: %s", model_out_path)


def evaluate_model(
    *,
    model: Any,
    test_bundle: DatasetBundle,
    threshold: float,
) -> dict[str, float]:
    """Evaluate binary classifier with thresholded predictions.

    Args:
        model: Trained model.
        test_bundle: Test dataset bundle.
        threshold: Threshold for converting probabilities to labels.

    Returns:
        Metric dictionary.
    """
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
    return metrics


def save_metrics(*, metrics: dict[str, float], model_type: str, threshold: float, output_csv: Path) -> None:
    """Save evaluation metrics to CSV.

    Args:
        metrics: Metric dictionary.
        model_type: Model type string.
        threshold: Decision threshold.
        output_csv: Output CSV path.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    row = {"model_type": model_type, "threshold": threshold}
    row.update(metrics)
    pd.DataFrame([row]).to_csv(output_csv, index=False)
    logging.info("Saved metrics: %s", output_csv)


def run_single_model(args: argparse.Namespace, *, model_type: str) -> None:
    """Run train/eval flow for one model type.

    Args:
        args: Parsed CLI arguments.
        model_type: lightgbm or svm.
    """
    model = build_model(
        model_type=model_type,
        random_state=args.random_state,
        svm_c=args.svm_c,
        svm_kernel=args.svm_kernel,
        svm_gamma=args.svm_gamma,
    )

    model_file = args.model_out_dir / f"{model_type}_{args.model_name}.joblib"

    if args.mode in {"train", "train_test"}:
        train_bundle = load_train_dataset(
            train_list_path=args.train_pdb_list,
            train_feature_dir=args.train_feature_dir,
            label_column=args.label_column,
        )
        print(f"Training {model_type} model with {len(train_bundle.labels)} samples and {train_bundle.features.shape[1]} features.")
        fit_and_save_model(
            model=model,
            train_bundle=train_bundle,
            model_out_path=model_file,
        )

    if args.mode in {"test", "train_test"}:
        test_bundle = load_test_dataset(
            test_list_path=args.test_pdb_list,
            test_feature_dir=args.test_feature_dir,
            test_prefix=args.test_prefix,
            label_column=args.label_column,
        )
        print(f"Evaluating {model_type} model with {len(test_bundle.labels)} samples and {test_bundle.features.shape[1]} features.")
        if args.mode == "test":
            model = load(model_file)

        metrics = evaluate_model(
            model=model,
            test_bundle=test_bundle,
            threshold=args.threshold,
        )
        metric_text = ", ".join([f"{key}={value:.4f}" for key, value in metrics.items()])
        logging.info("%s evaluation: %s", model_type, metric_text)

        metric_file = args.metrics_out_dir / f"{model_type}_{args.model_name}_metrics.csv"
        save_metrics(
            metrics=metrics,
            model_type=model_type,
            threshold=args.threshold,
            output_csv=metric_file,
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train/evaluate LightGBM and SVM with existing LigMet RF feature CSV.GZ files."
    )
    parser.add_argument("--mode", type=str, default="train_test", choices=["train", "test", "train_test"])
    parser.add_argument("--model_type", type=str, default="all", choices=["lightgbm", "svm", "all"])
    parser.add_argument("--model_name", type=str, default="baseline")

    parser.add_argument(
        "--label_column",
        type=str,
        default="label_2.0",
    )
    parser.add_argument(
        "--train_pdb_list",
        type=Path,
        default=Path("/home/qkrgangeun/LigMet/code/text/biolip/filtered/train_pdbs_chain_1_filtered.txt"),
    )
    parser.add_argument(
        "--train_feature_dir",
        type=Path,
        default=Path("/home/qkrgangeun/LigMet/data/biolip_backup/rf/features"),
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
    parser.add_argument("--test_prefix", type=str, default="")

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--svm_c", type=float, default=1.0)
    parser.add_argument("--svm_kernel", type=str, default="rbf", choices=["linear", "rbf", "poly", "sigmoid"])
    parser.add_argument("--svm_gamma", type=str, default="scale")

    parser.add_argument(
        "--model_out_dir",
        type=Path,
        default=Path("/home/qkrgangeun/LigMet/data/rf_param")
    )
    parser.add_argument(
        "--metrics_out_dir",
        type=Path,
        default=Path("/home/qkrgangeun/LigMet/data/rf_param/metrics"),
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main(*, args: argparse.Namespace) -> None:
    """Run baseline model training and evaluation.

    Args:
        args: Parsed command-line arguments.
    """
    setup_logging(verbose=args.verbose)

    if args.model_type == "all":
        model_types = ["lightgbm", "svm"]
    else:
        model_types = [args.model_type]

    for model_type in model_types:
        logging.info("Running model: %s", model_type)
        run_single_model(args=args, model_type=model_type)


if __name__ == "__main__":
    cli_args = parse_args()
    main(args=cli_args)
