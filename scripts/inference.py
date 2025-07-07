#!/usr/bin/env python
# inference.py
"""
End-to-end inference CLI for LigMet:
  dl:  .pdb → .npz
  rf:  .npz → .csv.gz → .npz
  test: Lightning DL model test via LightningCLI
All functions work with string paths; no use of pathlib.Path.
Defaults are loaded from config.yaml, and only string manipulation is used when deriving output locations.
"""
import argparse
import yaml
import os
import numpy as np
import torch
from lightning.pytorch.cli import LightningCLI

from ligmet.pipeline import extract_dl_features, extract_rf_features, run_rf_inference
from ligmet.featurizer import optimize_dtype
from ligmet.pl import LigMetModel, LigMetDataModule

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../configs/config.yaml")
with open(CONFIG_PATH, 'r') as f:
    cfg = yaml.safe_load(f)

pre = cfg['data']['preprocessed']
ont = cfg['data']['onthefly']
FEATURES_DIR   = pre['features_dir']
RF_OUTPUT_DIR  = pre['rf_result_dir']
DL_TEST_DIR    = cfg['data']['dl_test_result_dir']
DEFAULT_MODEL  = ont['rf_model']


def run_dl(input_pdb: str, output_npz: str):
    """
    Compute DL features from a PDB file and save to .npz.
    Args:
      input_pdb: string path to the .pdb file
      output_npz: string path where .npz will be written
    """
    feat_dict = extract_dl_features(input_pdb)
    compressed = {k: optimize_dtype(k, v) for k, v in feat_dict.items()}
    np.savez(output_npz, **compressed)
    print(f"[dl] saved DL features → {output_npz}")


def run_rf(input_npz: str, output_csv: str, output_npz: str, model_path: str):
    """
    Generate RF features and run RF inference.
    Args:
      input_npz: string path to DL features .npz
      output_csv: string path where RF features .csv.gz is written
      output_npz: string path where RF inference .npz is written
      model_path: string path to RF model (.joblib)
    """
    df = extract_rf_features(input_npz)
    df.to_csv(output_csv, index=False, compression='gzip')
    print(f"[rf-make] saved RF features → {output_csv}")
    probs = run_rf_inference(output_csv, model_path)
    np.savez(output_npz, prob=probs)
    print(f"[rf-test] saved RF inference → {output_npz}")


def main():
    parser = argparse.ArgumentParser("inference.py")
    parser.add_argument(
        "--pdb_path", type=str,
        help="Path to the PDB file"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str,
        help="Base output directory for all artifacts"
    )
    parser.add_argument(
        "--rf_model", "-m", type=str,
        default=None,
        help="Path to RF model (.joblib)"
    )
    parser.add_argument(
        "--cpu_only", action="store_true",
        help="Skip DL-model test step"
    )
    args = parser.parse_args()

    if args.cpu_only:
        pdb_path = args.pdb_path
        pdb_id = os.path.basename(pdb_path).replace('.pdb', '')
        base_out = args.output_dir 
        if base_out:
            dl_dir = f"{base_out}/dl/features"
            rf_csv_dir = f"{base_out}/rf/features"
            rf_npz_dir = f"{base_out}/rf/grid_prob"
            os.makedirs(dl_dir, exist_ok=True)
            os.makedirs(rf_csv_dir, exist_ok=True)
            os.makedirs(rf_npz_dir, exist_ok=True)
            # derive all paths as strings
            dl_out      = f"{base_out}/dl/features/{pdb_id}.npz"
            rf_csv_out  = f"{base_out}/rf/features/{pdb_id}.csv.gz" 
            rf_npz_out  = f"{base_out}/rf/grid_prob/{pdb_id}.npz" 
            model_path  = args.rf_model 
            dl_test_out = f"{base_out}/dl_test/{pdb_id}"
        else:
            dl_out      = FEATURES_DIR
            rf_csv_out  = f"{base_out}/rf/features/{pdb_id}.csv.gz" 
            rf_npz_out  = RF_OUTPUT_DIR
            model_path  = DEFAULT_MODEL
            dl_test_out = DL_TEST_DIR

        # Step 1: DL features
        run_dl(pdb_path, dl_out)
        # Step 2: RF pipeline
        run_rf(dl_out, rf_csv_out, rf_npz_out, model_path)

    # Step 3: DL-model test via LightningCLI (optional)
    if not args.cpu_only:
        cli = LightningCLI(
            LigMetModel,
            LigMetDataModule,
            save_config_kwargs={"overwrite": True},
            run=False
        )
        cli.trainer.test(
            cli.model,
            datamodule=cli.datamodule
        )

if __name__ == '__main__':
    main()
