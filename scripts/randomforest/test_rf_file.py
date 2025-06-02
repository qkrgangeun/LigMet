import argparse
from joblib import load
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import os

def load_data(test_data_path, label_column='label_2.0'):
    """ Load and preprocess training and testing data """
    X_test_list = []
    Y_test_list = []

    test_data = [pdb_id.strip() for pdb_id in open(test_data_path, 'r')]
    for pdb_id in test_data:
        data_path = f"/home/qkrgangeun/LigMet/data/biolip/rf/features/{pdb_id}.csv.gz"
        data = pd.read_csv(data_path, compression='gzip')  # Read the corresponding .csv.gz file without extracting

        # Drop the label column and append to features
        X_test_list.append(data.drop([label_column], axis=1))
        Y_test_list.append(data[label_column])
    print('--test data loaded')

    X_test = pd.concat(X_test_list, ignore_index=True)
    Y_test = pd.concat(Y_test_list, ignore_index=True)

    return X_test, Y_test

def main():
    print('!')
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file (.joblib)")
    parser.add_argument("--test_data", type=str, required=True, help="Path to a text file containing PDB IDs, one per line")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")

    args = parser.parse_args()

    # Load model and test data
    print("-> Loading model and test data")
    model = load(args.model_path)
    X_test, Y_test = load_data(args.test_data)

    # Prediction
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= args.threshold).astype(int)

    # Evaluation
    accuracy = accuracy_score(Y_test, y_pred)
    report = classification_report(Y_test, y_pred)
    print(f"Threshold: {args.threshold}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", report)

    true_count = sum(y_pred == 1)
    print(f"Total grids: {len(y_pred)}")
    print(f"True predicted grids: {true_count}")
    print(f"True ratio: {true_count / len(y_pred):.4f}")
    
if __name__ == "__main__":
    print('main')
    main()