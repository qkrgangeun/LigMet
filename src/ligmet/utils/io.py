import numpy as np
import pandas as pd

def load_npz(path):
    # 빈 metal_positions/types 자동 생성
    raw = dict(np.load(path, allow_pickle=True))
    raw.setdefault("metal_positions", np.empty((0,3)))
    raw.setdefault("metal_types",    np.empty((0,)))
    return raw

def save_csv_as_gzip(df, out_path):
    df.to_csv(out_path, index=False, compression="gzip")

def load_features_csv_gzip(pdb_id, feat_dir, label_col="label_2.0"):
    path = feat_dir / f"{pdb_id}.csv.gz"
    df = pd.read_csv(path, compression="gzip")
    return df.drop([label_col], axis=1), df[label_col]
