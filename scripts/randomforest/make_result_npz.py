import argparse
from joblib import load
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import os

def load_data(pdb_id, label_column='label_2.0'):
    """테스트 데이터 로드 및 전처리"""
    data_path = f"/home/qkrgangeun/LigMet/data/biolip/rf/features/{pdb_id}.csv.gz"
    data = pd.read_csv(data_path, compression='gzip')
    X = data.drop([label_column], axis=1)
    Y = data[label_column]
    return X, Y

def save_predicted_grids(pdb_id, y_pred, output_dir):
    """True로 예측된 grid의 위치를 .xyz 파일로 저장"""
    npz_path = f"/home/qkrgangeun/LigMet/data/biolip/dl/features/{pdb_id}.npz"
    df = np.load(npz_path)
    grid_positions = df["grid_positions"]
    true_grid_positions = grid_positions[y_pred == 1]

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{pdb_id}_grid.xyz")

    with open(output_path, 'w') as file:
        for i, coord in enumerate(true_grid_positions, start=1):
            # PDB ATOM 레코드 형식: ATOM, serial, name, resName, chainID, resSeq, x, y, z, occupancy, tempFactor, element
            line = (
                f"ATOM  {i:5d}  H   LIG A   1    "
                f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                f"  1.00  0.00           H\n"
            )
            file.write(line)
    print(f"Predicted grid saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file (.joblib)")
    parser.add_argument("--test_data", type=str, required=True, help="Single PDB ID (e.g., '1abc')")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold") 
    parser.add_argument("--output_dir", type=str, default='/home/qkrgangeun/LigMet/data/biolip/rf/grid_prob', help="Classification threshold")
    
    args = parser.parse_args()
    pdb_id = args.test_data

    # Load model and test data
    print("-> Loading model and test data")
    model = load(args.model_path)
    X_test, Y_test = load_data(pdb_id)

    # Prediction
    y_prob = model.predict_proba(X_test)[:, 1]
    output_dir = args.output_dir
    output_path = f"{output_dir}/{pdb_id}.npz"
    np.savez(output_path, prob=y_prob)
    print(output_path)
if __name__ == "__main__":
    main()
