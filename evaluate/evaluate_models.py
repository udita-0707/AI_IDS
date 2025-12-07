#!/usr/bin/env python3
"""
Evaluate saved models on provided test CSVs and print metrics.
Usage:
  python evaluate_models.py --binary_test final_preprocessed_data/csv/binary_test.csv \
                            --multi_test final_preprocessed_data/csv/multi_test.csv \
                            --anomaly_test final_preprocessed_data/csv/anomaly_test.csv
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from utils.preprocessing import load_preprocessing_artifacts

MODELS_DIR = Path("final_preprocessed_data/models")

def load_models():
    with open(MODELS_DIR / "xgb_binary.pkl", "rb") as f:
        xgb_bin = pickle.load(f)
    with open(MODELS_DIR / "xgb_multiclass.pkl", "rb") as f:
        xgb_multi = pickle.load(f)
    try:
        import tensorflow as tf
        from tensorflow import keras
        ae = keras.models.load_model(str(MODELS_DIR / "autoencoder_anomaly_only.h5"), compile=False)
    except Exception:
        ae = None
    thr = None
    thr_path = MODELS_DIR / "reconstruction_threshold.pkl"
    if thr_path.exists():
        with open(thr_path, "rb") as f:
            thr = pickle.load(f)
    return xgb_bin, xgb_multi, ae, thr

def evaluate_binary(xgb_bin, artifacts, test_csv):
    print("[*] Evaluating Binary XGBoost")
    df = pd.read_csv(test_csv)
    y_true = df['Binary_Label'].values
    X = df.drop(columns=['Binary_Label']).astype(float)
    # preprocessing (simple: align to feature_names, scale, cluster, pca)
    X_dict, _, _ = preprocess_for_models(X, artifacts)
    X_bin = X_dict['binary']
    y_pred = xgb_bin.predict(X_bin)
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1:", f1_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

def evaluate_multiclass(xgb_multi, artifacts, test_csv):
    print("[*] Evaluating Multiclass XGBoost")
    df = pd.read_csv(test_csv)
    y_true = df['Label'].values
    X = df.drop(columns=['Label']).astype(float)
    X_dict, _, _ = preprocess_for_models(X, artifacts)
    X_multi = X_dict['multiclass']
    y_pred = xgb_multi.predict(X_multi)
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Macro F1:", f1_score(y_true, y_pred, average='macro'))
    print(classification_report(y_true, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

def evaluate_autoencoder(ae, thr, artifacts, test_csv):
    print("[*] Evaluating Autoencoder Anomaly Detector")
    df = pd.read_csv(test_csv)
    if 'Label' not in df.columns:
        print("Anomaly test CSV must have 'Label' (0 normal, 1 attack)")
        return
    y_true = df['Label'].values
    X = df.drop(columns=['Label']).astype(float)
    X_dict, _, _ = preprocess_for_models(X, artifacts)
    X_anom = X_dict['anomaly']
    if ae is None:
        print("Autoencoder not available.")
        return
    X_recon = ae.predict(X_anom, verbose=0, batch_size=256)
    mse = np.mean((X_anom - X_recon)**2, axis=1)
    if thr is None:
        thr_used = mse.mean() + 3*mse.std()
    else:
        thr_used = thr
    y_pred = (mse > thr_used).astype(int)
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1:", f1_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary_test", type=str, default=None)
    parser.add_argument("--multi_test", type=str, default=None)
    parser.add_argument("--anomaly_test", type=str, default=None)
    args = parser.parse_args()

    artifacts = load_preprocessing_artifacts()
    xgb_bin, xgb_multi, ae, thr = load_models()

    if args.binary_test:
        evaluate_binary(xgb_bin, artifacts, args.binary_test)
    if args.multi_test:
        evaluate_multiclass(xgb_multi, artifacts, args.multi_test)
    if args.anomaly_test:
        evaluate_autoencoder(ae, thr, artifacts, args.anomaly_test)
