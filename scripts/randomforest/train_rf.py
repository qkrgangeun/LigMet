import argparse
import os
import numpy as np
import pandas as pd
from joblib import dump, load
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path 
def load_data(train_data_path, test_data_path, label_column='label_2.0'):
    """ Load and preprocess training and testing data """
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    X_train = train_data.drop([label_column], axis=1)
    Y_train = train_data[label_column]
    X_test = test_data.drop([label_column], axis=1)
    Y_test = test_data[label_column]

    return X_train, Y_train, X_test, Y_test

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

def main():
    """ Main function to handle argument parsing and run training/evaluation """
    parser = argparse.ArgumentParser(description="Train and evaluate a Balanced Random Forest model")
    parser.add_argument("--model_name", type=str, required=True, help="Path to save the trained model")
    parser.add_argument("--train_data", type=str, required=True, help="Path to the training data CSV file")
    parser.add_argument("--test_data", type=str, required=True, help="Path to the testing data CSV file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for classification")

    args = parser.parse_args()

    # 데이터 로드
    X_train, Y_train, X_test, Y_test = load_data(args.train_data, args.test_data)

    # 모델 학습
    model_dir = '/home/qkrgangeun/LigMet/data/rf_param'
    model_path = Path(model_dir)/args.model_name
    os.makedirs(model_dir,exist_ok=True)
    model = train(model_path, X_train, Y_train)

    # 평가
    evaluate(model, X_test, Y_test, args.threshold)

if __name__ == "__main__":
    main()
