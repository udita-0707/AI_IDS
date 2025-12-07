#!/usr/bin/env python3
"""
live_predict.py
Full inference pipeline that:
 - Accepts a PCAP file OR a flows CSV
 - Extracts flows using CICFlowMeter (via utils.flow_extractor)
 - Preprocesses flows to training format (utils.preprocessing)
 - Runs Binary & Multiclass XGBoost (pickle) and Autoencoder (h5)
 - Produces final CSV with columns:
     Flow_ID, Source_IP, Destination_IP, Source_Port, Destination_Port,
     Protocol, Timestamp, Label (BENIGN/ATTACK/ANOMALY), Attack_Type
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import os
import sys
import time

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    
# TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# local utils
from utils.flow_extractor import extract_flows_from_pcap
from utils.preprocessing import load_preprocessing_artifacts, preprocess_for_models

# Paths
BASE_DIR = Path("final_preprocessed_data")
MODELS_DIR = BASE_DIR / "models"
DEFAULT_OUTPUT = Path("live/live_predictions.csv")

# Attack names mapping
ATTACK_NAMES = {
    0: "Bot",
    1: "DDoS",
    2: "DoS GoldenEye",
    3: "DoS Hulk",
    4: "DoS Slowhttptest",
    5: "DoS slowloris",
    6: "FTP-Patator",
    7: "Heartbleed",
    8: "Infiltration",
    9: "PortScan",
    10: "SSH-Patator",
    11: "Web Attack - Brute Force",
    12: "Web Attack - Sql Injection",
    13: "Web Attack - XSS"
}

# ---- DECISION THRESHOLDS ----
# High-confidence attack for which we trust multiclass
BIN_ATTACK_CONF_THRESHOLD = 0.60

# AE threshold sensitivity: mean + k * std (smaller k => more anomalies)
# NOTE: This is only used if saved threshold is not available
# For better detection, we'll use a more sensitive multiplier on the saved threshold
ANOMALY_STD_MULTIPLIER = 0.5   # very sensitive (fallback only)

# Sensitivity factor for saved threshold (multiply saved threshold by this to make it more sensitive)
# Values < 1.0 make threshold more sensitive (detect more anomalies)
ANOMALY_THRESHOLD_SENSITIVITY = 0.7  # Use 70% of saved threshold = more sensitive

# We will NOT gate anomalies on confidence anymore, but keep this for display
ANOMALY_CONF_THRESHOLD = 0.0


def load_models(models_dir=MODELS_DIR):
    models = {}
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")
        
        with open(models_dir / "xgb_binary.pkl", "rb") as f:
            models["binary"] = pickle.load(f)
        with open(models_dir / "xgb_multiclass.pkl", "rb") as f:
            models["multiclass"] = pickle.load(f)
    
    # Debug: show binary classes_ so we know what index is ATTACK vs BENIGN
    if hasattr(models["binary"], "classes_"):
        print(f"[+] Binary model classes_: {models['binary'].classes_}")
    else:
        print("[!] Binary model has no 'classes_' attribute; using default indices")

    auto_path = models_dir / "autoencoder_anomaly_only.h5"
    if TF_AVAILABLE and auto_path.exists():
        try:
            models["autoencoder"] = keras.models.load_model(str(auto_path), compile=False)
            print(f"[+] Loaded autoencoder from: {auto_path}")
        except Exception as e:
            print(f"[!] Error loading autoencoder: {e}")
            models["autoencoder"] = None
    else:
        if not TF_AVAILABLE:
            print("[!] TensorFlow not available - anomaly detection disabled")
        if not auto_path.exists():
            print(f"[!] Autoencoder not found at: {auto_path}")
        models["autoencoder"] = None
    
    thr_path = models_dir / "reconstruction_threshold.pkl"
    if thr_path.exists():
        with open(models_dir / "reconstruction_threshold.pkl", "rb") as f:
            models["threshold"] = pickle.load(f)
        print(f"[+] Loaded reconstruction threshold: {models['threshold']:.6f}")
    else:
        print("[!] Reconstruction threshold not found - will use adaptive threshold")
        models["threshold"] = None

    return models


def _get_attack_benign_indices(binary_model):
    """
    Infer which index in predict_proba corresponds to ATTACK vs BENIGN.
    Handles:
      - classes_ = ['BENIGN', 'ATTACK']
      - classes_ = ['ATTACK', 'BENIGN']
      - classes_ = [0, 1] or [1, 0]
    Falls back to (benign=0, attack=1) if unknown.
    """
    attack_idx = 1
    benign_idx = 0

    if not hasattr(binary_model, "classes_"):
        print("[!] Binary model has no classes_; assuming index 1 = ATTACK, 0 = BENIGN")
        return attack_idx, benign_idx

    classes = binary_model.classes_
    print(f"[DEBUG] Binary classes_: {classes}")

    # String labels
    if ("ATTACK" in classes) and ("BENIGN" in classes):
        attack_idx = int(np.where(classes == "ATTACK")[0][0])
        benign_idx = int(np.where(classes == "BENIGN")[0][0])
        print(f"[+] Detected string classes: ATTACK index={attack_idx}, BENIGN index={benign_idx}")
    # Integer labels
    elif (1 in classes) and (0 in classes):
        attack_idx = int(np.where(classes == 1)[0][0])
        benign_idx = int(np.where(classes == 0)[0][0])
        print(f"[+] Detected int classes: ATTACK index={attack_idx}, BENIGN index={benign_idx}")
    else:
        print("[!] Could not infer ATTACK/BENIGN indices from classes_. "
              "Defaulting to ATTACK=1, BENIGN=0. Please verify your model labels.")

    return attack_idx, benign_idx


def run_pipeline(input_path, output_csv, is_pcap=True, debug=False):
    print("=" * 70)
    print("AI_IDS INFERENCE PIPELINE (SEQUENTIAL LOGIC)")
    print("=" * 70)
    
    # 1) Extract flows if pcap, else load CSV
    if is_pcap:
        print(f"[+] Extracting flows from PCAP: {input_path}")
        flows_csv_path = Path("live") / "live_flows.csv"
        flows_csv_path.parent.mkdir(parents=True, exist_ok=True)
        extract_flows_from_pcap(str(input_path), str(flows_csv_path))
        df_raw = pd.read_csv(flows_csv_path, low_memory=False)
    else:
        print(f"[+] Loading flows from CSV: {input_path}")
        df_raw = pd.read_csv(input_path, low_memory=False)

    print(f"[+] Loaded {len(df_raw)} flows")
    
    # PRESERVE ORIGINAL METADATA BEFORE PREPROCESSING
    metadata_cols = {}
    
    # Try multiple possible column names for each field
    ip_src_cols = ['Source IP', 'src_ip', 'Src IP', 'source_ip', 'Source_IP']
    ip_dst_cols = ['Destination IP', 'dst_ip', 'Dst IP', 'destination_ip', 'Destination_IP']
    port_src_cols = ['Source Port', 'src_port', 'Src Port', 'source_port', 'Source_Port']
    port_dst_cols = ['Destination Port', 'dst_port', 'Dst Port', 'destination_port', 'Destination_Port']
    proto_cols = ['Protocol', 'protocol', 'Proto']
    time_cols = ['Timestamp', 'timestamp', 'Time']
    
    # Find and store the actual columns
    for col in df_raw.columns:
        if col in ip_src_cols:
            metadata_cols['src_ip'] = df_raw[col].copy()
        elif col in ip_dst_cols:
            metadata_cols['dst_ip'] = df_raw[col].copy()
        elif col in port_src_cols:
            metadata_cols['src_port'] = df_raw[col].copy()
        elif col in port_dst_cols:
            metadata_cols['dst_port'] = df_raw[col].copy()
        elif col in proto_cols:
            metadata_cols['protocol'] = df_raw[col].copy()
        elif col in time_cols:
            metadata_cols['timestamp'] = df_raw[col].copy()
    
    print(f"[+] Preserved metadata columns: {list(metadata_cols.keys())}")

    # 2) Load artifacts
    artifacts = load_preprocessing_artifacts(models_dir=MODELS_DIR)

    # 3) Preprocess for models
    X_dict, valid_indices, df_aligned = preprocess_for_models(df_raw, artifacts)

    # 4) Load models
    models = load_models(MODELS_DIR)

    # 5) Binary prediction - ALWAYS RUN (GATEKEEPER)
    print("\n[STAGE 1] Running Binary Classification...")
    X_bin = X_dict['binary']
    bin_proba = (
        models['binary'].predict_proba(X_bin)
        if hasattr(models['binary'], "predict_proba")
        else models['binary'].predict(X_bin)
    )

    # Ensure 2D shape
    if bin_proba.ndim == 1:
        prob_attack = bin_proba
        prob_benign = 1 - prob_attack
        bin_proba = np.vstack([prob_benign, prob_attack]).T

    # detect which column is ATTACK vs BENIGN
    attack_idx, benign_idx = _get_attack_benign_indices(models['binary'])

    bin_pred = np.argmax(bin_proba, axis=1)
    bin_conf = bin_proba[:, attack_idx]  # confidence that this flow is ATTACK

    # Human-readable raw binary label
    binary_raw_label_array = np.where(bin_pred == attack_idx, "ATTACK", "BENIGN")
    
    bin_attacks = (binary_raw_label_array == "ATTACK").sum()
    bin_benign = (binary_raw_label_array == "BENIGN").sum()
    print(f"  Binary Results: {bin_attacks} ATTACK, {bin_benign} BENIGN")

    # 6) Multiclass prediction - ONLY RUN FOR HIGH-CONFIDENCE ATTACKS
    print("\n[STAGE 2] Running Multiclass Classification (for attacks)...")
    X_multi = X_dict['multiclass']
    
    multi_pred = np.full(len(X_multi), -1, dtype=int)
    multi_proba = np.zeros(len(X_multi), dtype=float)
    
    high_conf_attack_mask = (binary_raw_label_array == "ATTACK") & (bin_conf >= BIN_ATTACK_CONF_THRESHOLD)
    attack_indices = np.where(high_conf_attack_mask)[0]
    
    if len(attack_indices) > 0:
        X_multi_subset = X_multi[attack_indices]
        multi_pred[attack_indices] = models['multiclass'].predict(X_multi_subset)
        if hasattr(models['multiclass'], "predict_proba"):
            multi_proba[attack_indices] = np.max(
                models['multiclass'].predict_proba(X_multi_subset), axis=1
            )
        print(f"  Multiclass ran on {len(attack_indices)} high-confidence attacks")
    else:
        print("  No high-confidence attacks - multiclass skipped")

    # 7) Autoencoder - RUN ON ALL FLOWS (not just non-high-conf attacks)
    # This ensures we catch anomalies even if binary classifier is wrong
    print("\n[STAGE 3] Running Anomaly Detection (on all flows)...")
    n_flows = X_bin.shape[0]
    anomaly_pred = np.zeros(n_flows, dtype=int)
    anomaly_conf = np.zeros(n_flows, dtype=float)
    mse_all = np.zeros(n_flows, dtype=float)

    if models['autoencoder'] is not None:
        try:
            X_anom_full = X_dict['anomaly']
            
            # Run anomaly detection on ALL flows, not just non-high-confidence attacks
            # This ensures we catch anomalies even if binary classifier mislabels them
            print(f"  Running anomaly detection on all {n_flows} flows...")

            X_recon = models['autoencoder'].predict(X_anom_full, verbose=0, batch_size=256)
            mse_all = np.mean((X_anom_full - X_recon) ** 2, axis=1)

            saved_thr = models.get('threshold', None)
            
            if saved_thr is None:
                # Fallback: calculate adaptive threshold (but this is less reliable)
                # Use a more sensitive multiplier
                adaptive_thr = mse_all.mean() + ANOMALY_STD_MULTIPLIER * mse_all.std()
                thr = adaptive_thr
                print(f"  [!] No saved threshold found - using adaptive threshold: {thr:.6f}")
                print(f"      (Mean MSE: {mse_all.mean():.6f}, Std: {mse_all.std():.6f})")
            else:
                # Use saved threshold but apply sensitivity factor to make it more sensitive
                # This makes the threshold lower, so more anomalies are detected
                thr = saved_thr * ANOMALY_THRESHOLD_SENSITIVITY
                print(f"  Using saved threshold with sensitivity factor:")
                print(f"      Saved threshold: {saved_thr:.6f}")
                print(f"      Sensitivity factor: {ANOMALY_THRESHOLD_SENSITIVITY}")
                print(f"      Adjusted threshold: {thr:.6f}")
                print(f"      (Mean MSE: {mse_all.mean():.6f}, Max MSE: {mse_all.max():.6f}, Min MSE: {mse_all.min():.6f})")
                print(f"      (Std MSE: {mse_all.std():.6f})")

            # Count how many flows exceed threshold
            for i in range(n_flows):
                is_anom = mse_all[i] > thr
                anomaly_pred[i] = 1 if is_anom else 0
                anomaly_conf[i] = float(
                    np.clip(mse_all[i] / (thr + 1e-8), 0, 1)
                )

            num_anomalies = anomaly_pred.sum()
            print(f"  ✓ Detected {num_anomalies} anomalies out of {n_flows} flows ({100*num_anomalies/n_flows:.1f}%)")
            
            # Debug: show distribution of MSE values
            if debug:
                print(f"  [DEBUG] MSE statistics:")
                print(f"    Min: {mse_all.min():.6f}")
                print(f"    25th percentile: {np.percentile(mse_all, 25):.6f}")
                print(f"    Median: {np.median(mse_all):.6f}")
                print(f"    75th percentile: {np.percentile(mse_all, 75):.6f}")
                print(f"    95th percentile: {np.percentile(mse_all, 95):.6f}")
                print(f"    Max: {mse_all.max():.6f}")
                print(f"    Threshold: {thr:.6f}")
                print(f"    Flows above threshold: {num_anomalies}")
                
        except Exception as e:
            print(f"[!] Error during anomaly detection: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("  Anomaly detection disabled (autoencoder not available)")

    # 8) Build final output - SEQUENTIAL DECISION LOGIC
    print("\n[STAGE 4] Applying Sequential Decision Logic...")
    results = []
    
    for i in range(n_flows):
        src_ip = metadata_cols.get('src_ip', pd.Series(["N/A"] * n_flows)).iloc[i]
        dst_ip = metadata_cols.get('dst_ip', pd.Series(["N/A"] * n_flows)).iloc[i]
        src_port = metadata_cols.get('src_port', pd.Series(["N/A"] * n_flows)).iloc[i]
        dst_port = metadata_cols.get('dst_port', pd.Series(["N/A"] * n_flows)).iloc[i]
        protocol = metadata_cols.get('protocol', pd.Series(["N/A"] * n_flows)).iloc[i]
        timestamp = metadata_cols.get('timestamp', pd.Series(["N/A"] * n_flows)).iloc[i]

        binary_raw_label = binary_raw_label_array[i]

        high_conf_attack = (
            binary_raw_label == "ATTACK" and
            bin_conf[i] >= BIN_ATTACK_CONF_THRESHOLD
        )

        has_attack_type = (multi_pred[i] != -1)

        # FINAL PRIORITY:
        # 1. High-conf + typed → ATTACK
        # 2. High-conf but no type → ANOMALY (UNKNOWN)
        # 3. If AE says anomaly → ANOMALY
        # 4. Else → BENIGN

        if high_conf_attack and has_attack_type:
            final_label = "ATTACK"
            attack_type = ATTACK_NAMES.get(int(multi_pred[i]), "Unknown")

        elif high_conf_attack and not has_attack_type:
            final_label = "ANOMALY"
            attack_type = "UNKNOWN"

        elif anomaly_pred[i] == 1:
            final_label = "ANOMALY"
            attack_type = "UNKNOWN"

        else:
            final_label = "BENIGN"
            attack_type = "N/A"

        results.append({
            "Flow_ID": f"Flow_{i+1}",
            "Source_IP": str(src_ip),
            "Destination_IP": str(dst_ip),
            "Source_Port": str(src_port),
            "Destination_Port": str(dst_port),
            "Protocol": str(protocol),
            "Timestamp": str(timestamp),
            "Label": final_label,
            "Attack_Type": attack_type,
            "Binary_Label": binary_raw_label,
            "Binary_Confidence": float(bin_conf[i]),
            "Anomaly_Detected": bool(anomaly_pred[i]),
            "Anomaly_Confidence": float(anomaly_conf[i]),
            "Reconstruction_MSE": float(mse_all[i]) if len(mse_all) > i else 0.0
        })

    df_out = pd.DataFrame(results)
    output_csv_path = Path(output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv_path, index=False)
    
    print("\n" + "=" * 70)
    print("PREDICTION SUMMARY")
    print("=" * 70)
    label_counts = df_out['Label'].value_counts()
    for label, count in label_counts.items():
        print(f"  {label}: {count} flows")
    
    if 'Attack_Type' in df_out.columns:
        attack_types = df_out[df_out['Label'] == 'ATTACK']['Attack_Type'].value_counts()
        if len(attack_types) > 0:
            print("\n  Attack Type Breakdown:")
            for attack, count in attack_types.items():
                print(f"    {attack}: {count}")
    
    print(f"\n[+] Saved predictions to: {output_csv_path}")
    print("=" * 70)
    
    return df_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcap", type=str, default=None, help="Input PCAP file")
    parser.add_argument("--csv", type=str, default=None, help="Input flows CSV")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUTPUT), help="Output CSV path")
    args = parser.parse_args()

    if args.pcap is None and args.csv is None:
        print("Provide --pcap <file> OR --csv <flows.csv>")
        sys.exit(1)

    if args.pcap:
        inp = Path(args.pcap)
        if not inp.exists():
            print(f"PCAP not found: {inp}")
            sys.exit(1)
        df = run_pipeline(inp, args.out, is_pcap=True, debug=args.debug)
    else:
        inp = Path(args.csv)
        if not inp.exists():
            print(f"CSV not found: {inp}")
            sys.exit(1)
        df = run_pipeline(inp, args.out, is_pcap=False, debug=args.debug)
