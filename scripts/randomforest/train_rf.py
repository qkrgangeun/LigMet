import argparse
import os
import numpy as np
import pandas as pd
from joblib import dump, load
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path 
from tqdm import tqdm
import gc
# def load_data(train_data_path, test_data_path, label_column='label_2.0'):
#     """ Load and preprocess training and testing data """
#     train_data = pd.read_csv(train_data_path)
#     test_data = pd.read_csv(test_data_path)

#     X_train = train_data.drop([label_column], axis=1)
#     Y_train = train_data[label_column]
#     X_test = test_data.drop([label_column], axis=1)
#     Y_test = test_data[label_column]

#     return X_train, Y_train, X_test, Y_test
def load_data(train_data_path, test_data_path, label_column='label_2.0'):
    """ Load and preprocess training and testing data """
    # Initialize lists to hold data
    X_train_list = []
    Y_train_list = []
    X_test_list = []
    Y_test_list = []

    # Read train data
    train_data = [pdb_id.strip() for pdb_id in open(train_data_path, 'r')]
    af_train_data = [pdb_id.strip() for pdb_id in open('/home/qkrgangeun/LigMet/data/biolip_backup/af2.3/train/af_trainset.txt', 'r')]

    for pdb_id in tqdm(train_data):
 # Assuming the 'pdb_id' column is in the train data
        data_path = f"/home/qkrgangeun/LigMet/data/biolip_backup/rf/features/{pdb_id}.csv.gz"
        data = pd.read_csv(data_path, compression='gzip')  # Read the corresponding .csv.gz file without extracting
        # Drop the label column and append to features
        X_train_list.append(data.drop([label_column], axis=1))
        Y_train_list.append(data[label_column])
        
    for af_pdb_id in tqdm(af_train_data):
        data_path = f"/home/qkrgangeun/LigMet/data/biolip_backup/af2.3/train/rf/features/AF_{af_pdb_id}.csv.gz"
        data = pd.read_csv(data_path, compression='gzip')
        X_train_list.append(data.drop([label_column], axis=1))
        Y_train_list.append(data[label_column])
    print('--train data loaded')
    # Read test data
    # test_data = [pdb_id.strip() for pdb_id in open(test_data_path, 'r')]
    # for pdb_id in test_data:
    #     data_path = f"/home/qkrgangeun/LigMet/data/biolip/rf/features/{pdb_id}.csv.gz"
    #     data = pd.read_csv(data_path, compression='gzip')  # Read the corresponding .csv.gz file without extracting

    #     # Drop the label column and append to features
    #     X_test_list.append(data.drop([label_column], axis=1))
    #     Y_test_list.append(data[label_column])
    # print('--test data loaded')
    # Concatenate all data into DataFrames
    X_train = pd.concat(X_train_list, ignore_index=True)
    Y_train = pd.concat(Y_train_list, ignore_index=True)
    # X_test = pd.concat(X_test_list, ignore_index=True)
    # Y_test = pd.concat(Y_test_list, ignore_index=True)
    X_test = []
    Y_test = []
    return X_train, Y_train, X_test, Y_test

def load_test_data(test_data_path, label_column='label_2.0'):
    """ Load and preprocess testing data """
    # Initialize lists to hold data
    X_test_list = []
    Y_test_list = []

    # Read test data
    test_data = [pdb_id.strip() for pdb_id in open(test_data_path, 'r')]
    for pdb_id in tqdm(test_data):
        data_path = f"/home/qkrgangeun/LigMet/data/biolip_backup/af2.3/testset_chain1/rf/features/{pdb_id}.csv.gz"
        data = pd.read_csv(data_path, compression='gzip')  # Read the corresponding .csv.gz file without extracting

        # Drop the label column and append to features
        X_test_list.append(data.drop([label_column], axis=1))
        Y_test_list.append(data[label_column])
    
    print('--test data loaded')
    
    # Concatenate all data into DataFrames
    X_test = pd.concat(X_test_list, ignore_index=True)
    Y_test = pd.concat(Y_test_list, ignore_index=True)

    return X_test, Y_test

def train(model_path, X_train, Y_train):
    """ Train a BalancedRandomForestClassifier and save the model """
    rf = BalancedRandomForestClassifier(random_state=42, n_jobs=-1)
    rf.fit(X_train, Y_train)

    # 모델 저장
    dump(rf, model_path)
    print(f'Model saved to {model_path}')
    
    return rf

def evaluate(model, X_test, y_test, threshold=0.5):
    """ Evaluate the model with the given threshold """
    # 예측 확률 계산
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # 평가 지표 출력
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f'Threshold: {threshold}')
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", report)

    # 특성 중요도 출력
    num_features = X_test.shape[1]
    print(f"Number of features used by the model: {num_features}")

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = X_test.columns

    print("Feature ranking:")
    for f in range(num_features):
        print(f"{f + 1}. feature {indices[f]} ({importances[indices[f]]:.4f}) - {feature_names[indices[f]]}")

    # True로 예측된 개수
    true_count = sum(y_pred == 1)
    print(f"Total grids: {len(y_pred)}") 
    print(f"True predicted grids: {true_count}") 
    print(f"True ratio: {true_count / len(y_pred):.4f}\n")
    
    return y_pred

def make_result(pdb_ids, feat_dir, out_dir, model_path):
    """
    각 pdb의 feature npz 파일을 읽어 예측 결과(prob 등)를 out_dir/pdbid.npz로 저장
    Args:
        pdb_ids (list): 저장할 PDB ID 리스트
        feat_dir (str): feature csv.gz.zip 파일 경로 
        out_dir (str): 저장 디렉토리
        model: 학습된 모델 path
    """
    os.makedirs(out_dir, exist_ok=True)
    model = load(model_path)
    
    for pdb_id in tqdm(pdb_ids):
        data_path = os.path.join(feat_dir, f"{pdb_id}.csv.gz")
        if not os.path.isfile(data_path):
            print(f"Feature not found: {data_path}")
            continue
        
        data = pd.read_csv(data_path, compression='gzip')
        data = data.drop(['label_2.0'], axis=1)
        prob = model.predict_proba(data)[:, 1]  # (n_samples,)
        np.savez(os.path.join(out_dir, f"{pdb_id}.npz"), prob=prob)
        print(f"Saved predictions for {pdb_id} to {out_dir}/{pdb_id}.npz")
        
def train():
    """ Main function to handle argument parsing and run training/evaluation """
    parser = argparse.ArgumentParser(description="Train and evaluate a Balanced Random Forest model")
    parser.add_argument("--model_name", type=str, required=True, help="Path to save the trained model")
    parser.add_argument("--train_data", type=str, default="/home/qkrgangeun/LigMet/code/text/biolip/filtered/train_pdbs_chain_1_filtered.txt", help="Path to the training data CSV file")
    parser.add_argument("--test_data", type=str, default="/home/qkrgangeun/LigMet/code/text/biolip/filtered/val_pdbs_filtered.txt", help="Path to the testing data CSV file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for classification")

    args = parser.parse_args()

    # 데이터 로드
    print("-> data loading start")
    X_train, Y_train, X_test, Y_test = load_data(args.train_data, args.test_data)
    print("-> data is successfully loaded")
    # 모델 학습
    model_dir = '/home/qkrgangeun/LigMet/data/rf_param'
    model_path = Path(model_dir)/args.model_name
    os.makedirs(model_dir,exist_ok=True)
    
    print('model path:',model_path)
    print('-> model train start')
    model = train(model_path, X_train, Y_train)
    del X_train, Y_train
    gc.collect()
    print('->model train finished')


def test():
    """ Main function to handle argument parsing and run evaluation """
    parser = argparse.ArgumentParser(description="Evaluate a Balanced Random Forest model")
    parser.add_argument("--model_name", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--test_data", type=str, default="/home/qkrgangeun/LigMet/data/biolip_backup/pdb/test_pdb_noerror.txt", help="Path to the testing data CSV file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for classification")

    args = parser.parse_args()

    # 데이터 로드
    print("-> data loading start")
    X_test, Y_test = load_test_data(args.test_data)
    print("-> data is successfully loaded")

    # 모델 로드
    model_path = Path(args.model_name)
    model = load(model_path)

    # 모델 평가
    print('-> model evaluation start')
    y_pred = evaluate(model, X_test, Y_test, args.threshold)
    print('-> model evaluation finished')
    

def results():
    """
    각 pdb의 feature npz/csv.gz 파일을 읽어 예측 결과(prob 등)를 out_dir/pdbid.npz로 저장.
    Argparse로
        --model_path (학습된 모델)
        --pdb_list (예측할 PDB 리스트 파일(txt))
        --feat_dir (feature 파일 디렉토리)
        --out_dir (예측 결과 저장 디렉토리)
    등을 받을 수 있음.
    """
    parser = argparse.ArgumentParser(description="Save per-PDB prediction results using trained model")
    parser.add_argument("--model_path", type=str, required=True, help="Trained model .joblib path")
    parser.add_argument("--pdblist_txt", type=str, required=True, help="PDB list text file (one per line)")
    parser.add_argument("--feat_dir", type=str, required=True, help="Feature files directory")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save output .npz files")
    args = parser.parse_args()

    # PDB 리스트 불러오기
    with open(args.pdblist_txt) as f:
        pdb_ids = [line.strip() for line in f if line.strip()]

    # make_result 호출
    make_result(pdb_ids, args.feat_dir, args.out_dir, args.model_path)

if __name__ == "__main__":
    test()
