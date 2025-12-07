"""
utils/preprocessing.py
Load preprocessing artifacts and provide preprocess_for_models(df, artifacts)
Handles multiple CSV formats: CICFlowMeter (lowercase_underscore) and space-separated formats
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd

MODELS_DIR = Path("final_preprocessed_data/models")

def create_cicflowmeter_to_training_mapping():
    """
    Map CICFlowMeter format (lowercase with underscores) to training format (Title_Case with underscores)
    'flow_duration' -> 'Flow_Duration'
    """
    mapping = {
        'flow_duration': 'Flow_Duration',
        'tot_fwd_pkts': 'Total_Fwd_Packets',
        'tot_bwd_pkts': 'Total_Backward_Packets',
        'totlen_fwd_pkts': 'Total_Length_of_Fwd_Packets',
        'totlen_bwd_pkts': 'Total_Length_of_Bwd_Packets',
        'fwd_pkt_len_max': 'Fwd_Packet_Length_Max',
        'fwd_pkt_len_min': 'Fwd_Packet_Length_Min',
        'fwd_pkt_len_mean': 'Fwd_Packet_Length_Mean',
        'fwd_pkt_len_std': 'Fwd_Packet_Length_Std',
        'bwd_pkt_len_max': 'Bwd_Packet_Length_Max',
        'bwd_pkt_len_min': 'Bwd_Packet_Length_Min',
        'bwd_pkt_len_mean': 'Bwd_Packet_Length_Mean',
        'bwd_pkt_len_std': 'Bwd_Packet_Length_Std',
        'flow_byts_s': 'Flow_Bytes/s',
        'flow_pkts_s': 'Flow_Packets/s',
        'flow_iat_mean': 'Flow_IAT_Mean',
        'flow_iat_std': 'Flow_IAT_Std',
        'flow_iat_max': 'Flow_IAT_Max',
        'flow_iat_min': 'Flow_IAT_Min',
        'fwd_iat_tot': 'Fwd_IAT_Total',
        'fwd_iat_mean': 'Fwd_IAT_Mean',
        'fwd_iat_std': 'Fwd_IAT_Std',
        'fwd_iat_max': 'Fwd_IAT_Max',
        'fwd_iat_min': 'Fwd_IAT_Min',
        'bwd_iat_tot': 'Bwd_IAT_Total',
        'bwd_iat_mean': 'Bwd_IAT_Mean',
        'bwd_iat_std': 'Bwd_IAT_Std',
        'bwd_iat_max': 'Bwd_IAT_Max',
        'bwd_iat_min': 'Bwd_IAT_Min',
        'fwd_psh_flags': 'Fwd_PSH_Flags',
        'bwd_psh_flags': 'Bwd_PSH_Flags',
        'fwd_urg_flags': 'Fwd_URG_Flags',
        'bwd_urg_flags': 'Bwd_URG_Flags',
        'fwd_header_len': 'Fwd_Header_Length',
        'bwd_header_len': 'Bwd_Header_Length',
        'fwd_pkts_s': 'Fwd_Packets/s',
        'bwd_pkts_s': 'Bwd_Packets/s',
        'pkt_len_max': 'Max_Packet_Length',
        'pkt_len_min': 'Min_Packet_Length',
        'pkt_len_mean': 'Packet_Length_Mean',
        'pkt_len_std': 'Packet_Length_Std',
        'pkt_len_var': 'Packet_Length_Variance',
        'fin_flag_cnt': 'FIN_Flag_Count',
        'syn_flag_cnt': 'SYN_Flag_Count',
        'rst_flag_cnt': 'RST_Flag_Count',
        'psh_flag_cnt': 'PSH_Flag_Count',
        'ack_flag_cnt': 'ACK_Flag_Count',
        'urg_flag_cnt': 'URG_Flag_Count',
        'cwe_flag_count': 'CWE_Flag_Count',
        'ece_flag_cnt': 'ECE_Flag_Count',
        'down_up_ratio': 'Down/Up_Ratio',
        'pkt_size_avg': 'Average_Packet_Size',
        'fwd_seg_size_avg': 'Avg_Fwd_Segment_Size',
        'bwd_seg_size_avg': 'Avg_Bwd_Segment_Size',
        'fwd_byts_b_avg': 'Fwd_Avg_Bytes/Bulk',
        'fwd_pkts_b_avg': 'Fwd_Avg_Packets/Bulk',
        'fwd_blk_rate_avg': 'Fwd_Avg_Bulk_Rate',
        'bwd_byts_b_avg': 'Bwd_Avg_Bytes/Bulk',
        'bwd_pkts_b_avg': 'Bwd_Avg_Packets/Bulk',
        'bwd_blk_rate_avg': 'Bwd_Avg_Bulk_Rate',
        'subflow_fwd_pkts': 'Subflow_Fwd_Packets',
        'subflow_fwd_byts': 'Subflow_Fwd_Bytes',
        'subflow_bwd_pkts': 'Subflow_Bwd_Packets',
        'subflow_bwd_byts': 'Subflow_Bwd_Bytes',
        'init_fwd_win_byts': 'Init_Win_bytes_forward',
        'init_bwd_win_byts': 'Init_Win_bytes_backward',
        'fwd_act_data_pkts': 'act_data_pkt_fwd',
        'fwd_seg_size_min': 'min_seg_size_forward',
        'active_mean': 'Active_Mean',
        'active_std': 'Active_Std',
        'active_max': 'Active_Max',
        'active_min': 'Active_Min',
        'idle_mean': 'Idle_Mean',
        'idle_std': 'Idle_Std',
        'idle_max': 'Idle_Max',
        'idle_min': 'Idle_Min',
    }
    return mapping

def create_space_to_underscore_mapping():
    """
    Map space-separated format to underscore format
    'Flow Duration' -> 'Flow_Duration'
    Note: Self-mappings (already in underscore format) are excluded
    """
    mapping = {
        'Flow Duration': 'Flow_Duration',
        'Total Fwd Packets': 'Total_Fwd_Packets',
        'Total Backward Packets': 'Total_Backward_Packets',
        'Total Length of Fwd Packets': 'Total_Length_of_Fwd_Packets',
        'Total Length of Bwd Packets': 'Total_Length_of_Bwd_Packets',
        'Fwd Packet Length Max': 'Fwd_Packet_Length_Max',
        'Fwd Packet Length Min': 'Fwd_Packet_Length_Min',
        'Fwd Packet Length Mean': 'Fwd_Packet_Length_Mean',
        'Fwd Packet Length Std': 'Fwd_Packet_Length_Std',
        'Bwd Packet Length Max': 'Bwd_Packet_Length_Max',
        'Bwd Packet Length Min': 'Bwd_Packet_Length_Min',
        'Bwd Packet Length Mean': 'Bwd_Packet_Length_Mean',
        'Bwd Packet Length Std': 'Bwd_Packet_Length_Std',
        'Flow Bytes/s': 'Flow_Bytes/s',
        'Flow Packets/s': 'Flow_Packets/s',
        'Flow IAT Mean': 'Flow_IAT_Mean',
        'Flow IAT Std': 'Flow_IAT_Std',
        'Flow IAT Max': 'Flow_IAT_Max',
        'Flow IAT Min': 'Flow_IAT_Min',
        'Fwd IAT Total': 'Fwd_IAT_Total',
        'Fwd IAT Mean': 'Fwd_IAT_Mean',
        'Fwd IAT Std': 'Fwd_IAT_Std',
        'Fwd IAT Max': 'Fwd_IAT_Max',
        'Fwd IAT Min': 'Fwd_IAT_Min',
        'Bwd IAT Total': 'Bwd_IAT_Total',
        'Bwd IAT Mean': 'Bwd_IAT_Mean',
        'Bwd IAT Std': 'Bwd_IAT_Std',
        'Bwd IAT Max': 'Bwd_IAT_Max',
        'Bwd IAT Min': 'Bwd_IAT_Min',
        'Fwd PSH Flags': 'Fwd_PSH_Flags',
        'Bwd PSH Flags': 'Bwd_PSH_Flags',
        'Fwd URG Flags': 'Fwd_URG_Flags',
        'Bwd URG Flags': 'Bwd_URG_Flags',
        'Fwd Header Length': 'Fwd_Header_Length',
        'Bwd Header Length': 'Bwd_Header_Length',
        'Fwd Packets/s': 'Fwd_Packets/s',
        'Bwd Packets/s': 'Bwd_Packets/s',
        'Min Packet Length': 'Min_Packet_Length',
        'Max Packet Length': 'Max_Packet_Length',
        'Packet Length Mean': 'Packet_Length_Mean',
        'Packet Length Std': 'Packet_Length_Std',
        'Packet Length Variance': 'Packet_Length_Variance',
        'FIN Flag Count': 'FIN_Flag_Count',
        'SYN Flag Count': 'SYN_Flag_Count',
        'RST Flag Count': 'RST_Flag_Count',
        'PSH Flag Count': 'PSH_Flag_Count',
        'ACK Flag Count': 'ACK_Flag_Count',
        'URG Flag Count': 'URG_Flag_Count',
        'CWE Flag Count': 'CWE_Flag_Count',
        'ECE Flag Count': 'ECE_Flag_Count',
        'Down/Up Ratio': 'Down/Up_Ratio',
        'Average Packet Size': 'Average_Packet_Size',
        'Avg Fwd Segment Size': 'Avg_Fwd_Segment_Size',
        'Avg Bwd Segment Size': 'Avg_Bwd_Segment_Size',
        'Fwd Avg Bytes/Bulk': 'Fwd_Avg_Bytes/Bulk',
        'Fwd Avg Packets/Bulk': 'Fwd_Avg_Packets/Bulk',
        'Fwd Avg Bulk Rate': 'Fwd_Avg_Bulk_Rate',
        'Bwd Avg Bytes/Bulk': 'Bwd_Avg_Bytes/Bulk',
        'Bwd Avg Packets/Bulk': 'Bwd_Avg_Packets/Bulk',
        'Bwd Avg Bulk Rate': 'Bwd_Avg_Bulk_Rate',
        'Subflow Fwd Packets': 'Subflow_Fwd_Packets',
        'Subflow Fwd Bytes': 'Subflow_Fwd_Bytes',
        'Subflow Bwd Packets': 'Subflow_Bwd_Packets',
        'Subflow Bwd Bytes': 'Subflow_Bwd_Bytes',
        'Init Fwd Win Bytes': 'Init_Win_bytes_forward',
        'Init Bwd Win Bytes': 'Init_Win_bytes_backward',
        'Fwd Act Data Pkts': 'act_data_pkt_fwd',
        'Fwd Seg Size Min': 'min_seg_size_forward',
        'Active Mean': 'Active_Mean',
        'Active Std': 'Active_Std',
        'Active Max': 'Active_Max',
        'Active Min': 'Active_Min',
        'Idle Mean': 'Idle_Mean',
        'Idle Std': 'Idle_Std',
        'Idle Max': 'Idle_Max',
        'Idle Min': 'Idle_Min',
        # NOTE: 'Fwd Header Length.1' is dropped separately, not mapped
    }
    return mapping

def load_preprocessing_artifacts(models_dir=MODELS_DIR):
    models_dir = Path(models_dir)
    artifacts = {}
    
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")
        
        with open(models_dir / "scaler.pkl", "rb") as f:
            artifacts['scaler'] = pickle.load(f)
        with open(models_dir / "kmeans_bin.pkl", "rb") as f:
            artifacts['kmeans_bin'] = pickle.load(f)
        with open(models_dir / "kmeans_multi.pkl", "rb") as f:
            artifacts['kmeans_multi'] = pickle.load(f)
        with open(models_dir / "pca_bin.pkl", "rb") as f:
            artifacts['pca_bin'] = pickle.load(f)
        with open(models_dir / "pca_multi.pkl", "rb") as f:
            artifacts['pca_multi'] = pickle.load(f)
    
    with open(models_dir / "feature_names.pkl", "rb") as f:
        artifacts['feature_names'] = pickle.load(f)
    return artifacts

def preprocess_for_models(df_raw, artifacts):
    """
    Aligns & preprocesses df_raw into X_pca for binary, multiclass and anomaly.
    """
    df = df_raw.copy()
    feature_names = artifacts['feature_names']
    
    # Drop problematic duplicate header column if present
    if "Fwd Header Length.1" in df.columns:
        print("[+] Dropping duplicate column 'Fwd Header Length.1'")
        df = df.drop(columns=["Fwd Header Length.1"])
    
    # Step 1: Map CICFlowMeter format (lowercase with underscores) to training format
    cicflowmeter_indicators = ['flow_duration', 'tot_fwd_pkts', 'fwd_pkt_len_max', 'flow_byts_s', 'totlen_fwd_pkts']
    has_cicflowmeter_format = any(col in cicflowmeter_indicators for col in df.columns)
    
    # Also check if we have lowercase underscore pattern (more robust)
    if not has_cicflowmeter_format:
        sample_cols = [col for col in df.columns if isinstance(col, str) and col.islower() and '_' in col]
        if len(sample_cols) > 5:
            has_cicflowmeter_format = True
            print(f"[+] Detected CICFlowMeter format by pattern (found {len(sample_cols)} lowercase_underscore columns)")
    
    if has_cicflowmeter_format:
        print("[+] Detected CICFlowMeter format - mapping to training format...")
        mapping = create_cicflowmeter_to_training_mapping()
        rename_dict = {cic_name: train_name for cic_name, train_name in mapping.items() if cic_name in df.columns}
        
        if rename_dict:
            df = df.rename(columns=rename_dict)
            print(f"[+] Mapped {len(rename_dict)} CICFlowMeter features to training format")
        else:
            print("[!] No CICFlowMeter mappings applied - check if columns match expected format")
    
    # Step 2: Map space-separated format to underscore format (if not already done)
    has_spaces = any(' ' in str(col) for col in df.columns)
    
    if has_spaces:
        print("[+] Detected space-separated format - mapping to underscore format...")
        mapping = create_space_to_underscore_mapping()
        rename_dict = {space_name: underscore_name for space_name, underscore_name in mapping.items() if space_name in df.columns}
        
        if rename_dict:
            df = df.rename(columns=rename_dict)
            print(f"[+] Mapped {len(rename_dict)} space-separated features")
        else:
            print("[!] No space-separated mappings applied - columns may already be correct")
    
    # Ensure there are no duplicate column names after renaming
    if df.columns.duplicated().any():
        dup_cols = df.columns[df.columns.duplicated()].tolist()
        print(f"[!] Found duplicated columns after mapping: {dup_cols} - keeping first occurrence")
        df = df.loc[:, ~df.columns.duplicated()]
    
    # Step 3: Drop metadata columns
    drop_cols = [
        "Flow ID", "Flow_ID", "flow_id",
        "Source IP", "Destination IP", "src_ip", "dst_ip",
        "Timestamp", "timestamp", 
        "Source Port", "Destination Port", "src_port", "dst_port",
        "Label", "label", "Protocol", "protocol"
    ]
    
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Step 4: Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Step 5: Handle inf/nan
    df = df.replace([np.inf, -np.inf], np.nan)
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        print(f"[!] Filling {nan_count} NaN/inf values with 0")
        df = df.fillna(0.0)

    # Step 6: Align with training features
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        print(f"[!] WARNING: {len(missing)} features missing, filling with zeros")
        if len(missing) < 10:
            print(f"[!] Missing features: {missing}")
        for c in missing:
            df[c] = 0.0
    
    df_aligned = df[feature_names].copy()
    
    # Check data quality - count non-zero values per column
    non_zero_cols = (df_aligned != 0).any(axis=0)
    non_zero_count = non_zero_cols.sum()
    print(f"[+] Aligned to {len(feature_names)} features, {len(df_aligned)} rows")
    print(f"[+] Non-zero features: {non_zero_count}/{len(feature_names)}")
    
    if non_zero_count < len(feature_names) * 0.3:
        print(f"[!] CRITICAL: Only {non_zero_count} features have data - check mapping!")
    
    # Step 7: Scale and transform
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        X_scaled = artifacts['scaler'].transform(df_aligned.values)
    
    cluster_bin = artifacts['kmeans_bin'].predict(X_scaled)
    cluster_multi = artifacts['kmeans_multi'].predict(X_scaled)
    
    X_embed_bin = np.hstack([X_scaled, cluster_bin.reshape(-1,1)])
    X_embed_multi = np.hstack([X_scaled, cluster_multi.reshape(-1,1)])
    
    X_pca_bin = artifacts['pca_bin'].transform(X_embed_bin).astype(np.float32)
    X_pca_multi = artifacts['pca_multi'].transform(X_embed_multi).astype(np.float32)
    
    return {
        'binary': X_pca_bin, 
        'multiclass': X_pca_multi, 
        'anomaly': X_pca_bin
    }, df_aligned.index.tolist(), df_aligned
