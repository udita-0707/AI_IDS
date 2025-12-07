# ========= FINAL CICIDS2017 PREPROCESSING PIPELINE (CPU SAFE) ==========
# FIXED VERSION - Correct execution order
# Includes:
# • Load & clean dataset
# • Binary + Multiclass + Anomaly labels
# • Benign downsampling (memory safe)
# • Scaling → scaler.pkl (with feature_names_in_)
# • Binary oversampling
# • Multiclass capping (NO oversampling)
# • Anomaly dataset (normal only)
# • KMeans clustering embedding
# • PCA (20 components) → pca_bin.pkl, pca_multi.pkl, pca_anomaly.pkl
# • Train/Test split saved as CSV
# • ALL visualizations saved in separate folders
# =======================================================================

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler


# ========= Create folder structure ==========
os.makedirs("final_preprocessed_data", exist_ok=True)
os.makedirs("final_preprocessed_data/models", exist_ok=True)
os.makedirs("final_preprocessed_data/csv", exist_ok=True)
os.makedirs("final_preprocessed_data/pca_visualizations", exist_ok=True)
os.makedirs("final_preprocessed_data/stats_visualizations", exist_ok=True)

print("Folders created successfully.")


# ===========================
# STEP 1: LOAD CSV FILES
# ===========================
folder = "TrafficLabelling"
files = glob.glob(os.path.join(folder, "*.csv"))

# Filter out the pcap-converted CSV files (keep only original CICIDS CSV files)
files = [f for f in files if not f.endswith('_full_features.csv') and not f.endswith('traffic-analysis-exercise.csv')]

df_list = []
print("\nLoading CSV files...\n")
for file in files:
    print("Loading:", file)
    df_list.append(pd.read_csv(file, encoding="ISO-8859-1", low_memory=False))

df = pd.concat(df_list, ignore_index=True)
print("\nDataset Combined Shape:", df.shape)


# ===========================
# STEP 2: CLEAN DATA
# ===========================
df.columns = df.columns.str.strip().str.replace(" ", "_")
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()
df = df.drop_duplicates()

print("After Cleaning:", df.shape)


# ===========================
# STEP 3: LABEL GENERATION
# ===========================
enc = LabelEncoder()
df["Multiclass_Label"] = enc.fit_transform(df["Label"])
df["Binary_Label"] = df["Label"].apply(lambda x: 0 if x.lower() == "benign" else 1)

# Save label encoder for future use
pickle.dump(enc, open("final_preprocessed_data/models/label_encoder.pkl", "wb"))
print("Label encoder saved.")

print("\nBinary Counts:\n", df["Binary_Label"].value_counts())
print("\nMulticlass Counts:\n", df["Multiclass_Label"].value_counts())


# ===========================
# STEP 4: BENIGN REDUCTION
# ===========================
benign_df = df[df["Binary_Label"] == 0]
attack_df = df[df["Binary_Label"] == 1]

target_benign = int(len(attack_df) * 1.2)
benign_df = benign_df.sample(n=target_benign, random_state=42)

df_reduced = pd.concat([benign_df, attack_df], ignore_index=True)
df_reduced = df_reduced.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nAfter Benign Reduction:", df_reduced.shape)


# ===========================
# STEP 5: NUMERIC FEATURES
# ===========================
# Keep as DataFrame (IMPORTANT: for feature_names_in_)
features_df = df_reduced.drop(["Label", "Binary_Label", "Multiclass_Label"], axis=1)
numeric_features = features_df.select_dtypes(include=["float64", "int64"])

feature_names = numeric_features.columns.tolist()
print(f"\nNumber of numeric features: {len(feature_names)}")

# X as DataFrame here
X_df = numeric_features


# ===========================
# STEP 6: SCALING (FIXED)
# ===========================
scaler = StandardScaler()

# Fit on DataFrame (so scaler.feature_names_in_ is available)
scaler.fit(X_df)

# Get scaled numpy array
X_scaled = scaler.transform(X_df)

# Save scaler and feature names
pickle.dump(scaler, open("final_preprocessed_data/models/scaler.pkl", "wb"))
pickle.dump(feature_names, open("final_preprocessed_data/models/feature_names.pkl", "wb"))

print("\nScaler saved → final_preprocessed_data/models/scaler.pkl")
print("Feature names saved → final_preprocessed_data/models/feature_names.pkl")


# ===========================
# STEP 7A: BINARY OVERSAMPLING
# ===========================
y_bin = df_reduced["Binary_Label"].values
oversampler = RandomOverSampler(random_state=42)
X_bin, y_bin = oversampler.fit_resample(X_scaled, y_bin)
print("\nBinary After Oversampling:", X_bin.shape)


# ===========================
# STEP 7B: MULTICLASS CAPPING
# ===========================
max_samples_per_class = 60000
y_multi = df_reduced["Multiclass_Label"].values

df_multi = pd.DataFrame(X_scaled)
df_multi["label"] = y_multi

chunks = []
for lbl, group in df_multi.groupby("label"):
    if len(group) > max_samples_per_class:
        group = group.sample(max_samples_per_class, random_state=42)
    chunks.append(group)

df_multi_bal = pd.concat(chunks, ignore_index=True).reset_index(drop=True)

X_multi = df_multi_bal.drop("label", axis=1).values
y_multi = df_multi_bal["label"].values

print("Multiclass After Balancing:", X_multi.shape)


# ===========================
# STEP 7C: ANOMALY DETECTION DATASET (NORMAL ONLY)
# ===========================
print("\n=== Creating Anomaly Detection Dataset ===")

# Extract only benign samples from the ORIGINAL scaled data
benign_mask = df_reduced["Binary_Label"] == 0
X_anomaly = X_scaled[benign_mask]
y_anomaly = np.zeros(len(X_anomaly))  # All zeros (normal)

print(f"Anomaly dataset (normal only): {X_anomaly.shape}")

# Optional: Sample if too large (keep 50k-100k normal samples)
if len(X_anomaly) > 80000:
    indices = np.random.choice(len(X_anomaly), 80000, replace=False)
    X_anomaly = X_anomaly[indices]
    y_anomaly = y_anomaly[indices]
    print(f"Sampled down to: {X_anomaly.shape}")


# ===========================
# STEP 8: KMEANS EMBEDDING
# ===========================
print("\n=== Applying KMeans Clustering ===")

# Binary KMeans
kmeans_bin = KMeans(n_clusters=6, random_state=42, n_init=10)
cluster_bin = kmeans_bin.fit_predict(X_bin)
X_bin_embed = np.hstack([X_bin, cluster_bin.reshape(-1, 1)])

# Multiclass KMeans
kmeans_multi = KMeans(n_clusters=10, random_state=42, n_init=10)
cluster_multi = kmeans_multi.fit_predict(X_multi)
X_multi_embed = np.hstack([X_multi, cluster_multi.reshape(-1, 1)])

# Anomaly KMeans (use binary model for consistency)
cluster_anomaly = kmeans_bin.predict(X_anomaly)
X_anomaly_embed = np.hstack([X_anomaly, cluster_anomaly.reshape(-1, 1)])

print("Clustering embedded for all datasets.")

# Save KMeans models
pickle.dump(kmeans_bin, open("final_preprocessed_data/models/kmeans_bin.pkl", "wb"))
pickle.dump(kmeans_multi, open("final_preprocessed_data/models/kmeans_multi.pkl", "wb"))
print("KMeans models saved.")

# IMPORTANT: Add cluster names
feature_names_bin = feature_names + ["cluster_bin"]
feature_names_multi = feature_names + ["cluster_multi"]
feature_names_anomaly = feature_names + ["cluster_anomaly"]


# ===========================
# STEP 9: PCA (FIXED: SAVE MODELS)
# ===========================
print("\n=== Applying PCA ===")

# Binary PCA
pca_bin = PCA(n_components=20, random_state=42)
X_bin_pca = pca_bin.fit_transform(X_bin_embed)

# Multiclass PCA
pca_multi = PCA(n_components=20, random_state=42)
X_multi_pca = pca_multi.fit_transform(X_multi_embed)

# Anomaly PCA (use binary PCA for consistency)
X_anomaly_pca = pca_bin.transform(X_anomaly_embed)

print("PCA applied to all datasets.")

# Save PCA models
pickle.dump(pca_bin, open("final_preprocessed_data/models/pca_bin.pkl", "wb"))
pickle.dump(pca_multi, open("final_preprocessed_data/models/pca_multi.pkl", "wb"))
print("PCA models saved → pca_bin.pkl, pca_multi.pkl")


# ===========================
# STEP 10: TRAIN/TEST SPLIT
# ===========================
print("\n=== Creating Train/Test Splits ===")

binary_cols = [f"PC{i+1}" for i in range(20)] + ["Binary_Label"]
multi_cols = [f"PC{i+1}" for i in range(20)] + ["Multiclass_Label"]
anomaly_cols = [f"PC{i+1}" for i in range(20)] + ["Label"]

# Binary split
Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    X_bin_pca, y_bin, test_size=0.2, random_state=42, stratify=y_bin
)

# Multiclass split
Xm_train, Xm_test, ym_train, ym_test = train_test_split(
    X_multi_pca, y_multi, test_size=0.2, random_state=42, stratify=y_multi
)

# Anomaly split (all labels are 0, so no stratify)
Xa_train, Xa_test, ya_train, ya_test = train_test_split(
    X_anomaly_pca, y_anomaly, test_size=0.2, random_state=42
)

# Save Binary CSVs
pd.DataFrame(np.column_stack([Xb_train, yb_train]), columns=binary_cols) \
    .to_csv("final_preprocessed_data/csv/binary_train.csv", index=False)

pd.DataFrame(np.column_stack([Xb_test, yb_test]), columns=binary_cols) \
    .to_csv("final_preprocessed_data/csv/binary_test.csv", index=False)

pd.DataFrame(np.column_stack([X_bin_pca, y_bin]), columns=binary_cols) \
    .to_csv("final_preprocessed_data/csv/binary_preprocessed.csv", index=False)

# Save Multiclass CSVs
pd.DataFrame(np.column_stack([Xm_train, ym_train]), columns=multi_cols) \
    .to_csv("final_preprocessed_data/csv/multiclass_train.csv", index=False)

pd.DataFrame(np.column_stack([Xm_test, ym_test]), columns=multi_cols) \
    .to_csv("final_preprocessed_data/csv/multiclass_test.csv", index=False)

pd.DataFrame(np.column_stack([X_multi_pca, y_multi]), columns=multi_cols) \
    .to_csv("final_preprocessed_data/csv/multiclass_preprocessed.csv", index=False)

# Save Anomaly CSVs
pd.DataFrame(np.column_stack([Xa_train, ya_train]), columns=anomaly_cols) \
    .to_csv("final_preprocessed_data/csv/anomaly_train.csv", index=False)

pd.DataFrame(np.column_stack([Xa_test, ya_test]), columns=anomaly_cols) \
    .to_csv("final_preprocessed_data/csv/anomaly_test.csv", index=False)

pd.DataFrame(np.column_stack([X_anomaly_pca, y_anomaly]), columns=anomaly_cols) \
    .to_csv("final_preprocessed_data/csv/anomaly_preprocessed.csv", index=False)

print("All CSVs saved successfully!")


# ===========================
# STEP 11: VISUALIZATIONS
# ===========================
print("\n=== Generating Visualizations ===")

# 1) Binary class distribution
plt.figure(figsize=(10, 6))
sns.countplot(x=y_bin)
plt.title("Binary Class Distribution (After Oversampling)")
plt.xlabel("Class (0=Benign, 1=Attack)")
plt.ylabel("Count")
plt.savefig("final_preprocessed_data/stats_visualizations/binary_class_dist.png", dpi=100)
plt.close()

# 2) Multiclass class distribution
plt.figure(figsize=(14, 6))
unique, counts = np.unique(y_multi, return_counts=True)
plt.bar(unique, counts)
plt.title("Multiclass Class Distribution (After Balancing)")
plt.xlabel("Attack Class")
plt.ylabel("Count")
plt.xticks(unique)
plt.savefig("final_preprocessed_data/stats_visualizations/multiclass_class_dist.png", dpi=100)
plt.close()

# 3) Anomaly dataset distribution (should be all zeros)
plt.figure(figsize=(10, 6))
sns.countplot(x=y_anomaly)
plt.title("Anomaly Detection Dataset (Normal Traffic Only)")
plt.xlabel("Label (0=Normal)")
plt.ylabel("Count")
plt.savefig("final_preprocessed_data/stats_visualizations/anomaly_class_dist.png", dpi=100)
plt.close()

# 4) Correlation heatmap
corr = pd.DataFrame(X_scaled, columns=feature_names).iloc[:, :20].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap (Top 20 Features)")
plt.tight_layout()
plt.savefig("final_preprocessed_data/stats_visualizations/corr_heatmap_top20.png", dpi=100)
plt.close()

# 5) PCA Variance Explained
plt.figure(figsize=(10, 5))
plt.plot(range(1, 21), pca_bin.explained_variance_ratio_, marker='o', label='Binary')
plt.plot(range(1, 21), pca_multi.explained_variance_ratio_, marker='s', label='Multiclass')
plt.xlabel("Principal Component")
plt.ylabel("Variance Explained")
plt.title("PCA Variance Explained")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("final_preprocessed_data/pca_visualizations/viz_pca_variance.png", dpi=100)
plt.close()

# 6) PCA Scatter Binary
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_bin_pca[:, 0], X_bin_pca[:, 1], c=y_bin, cmap="coolwarm", s=3, alpha=0.5)
plt.colorbar(scatter, label="Class")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Scatter - Binary Classification")
plt.savefig("final_preprocessed_data/pca_visualizations/viz_pca_scatter_binary.png", dpi=100)
plt.close()

# 7) PCA Scatter Multiclass
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_multi_pca[:, 0], X_multi_pca[:, 1], c=y_multi, cmap="tab10", s=3, alpha=0.5)
plt.colorbar(scatter, label="Attack Type")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Scatter - Multiclass Classification")
plt.savefig("final_preprocessed_data/pca_visualizations/viz_pca_scatter_multiclass.png", dpi=100)
plt.close()

# 8) PCA Scatter Anomaly
plt.figure(figsize=(10, 7))
plt.scatter(X_anomaly_pca[:, 0], X_anomaly_pca[:, 1], c='green', s=3, alpha=0.5)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Scatter - Anomaly Detection (Normal Traffic)")
plt.savefig("final_preprocessed_data/pca_visualizations/viz_pca_scatter_anomaly.png", dpi=100)
plt.close()

# 9) PCA Feature Contribution for PC1-PC20
loadings = pca_bin.components_
top_n = 12

for comp in range(20):
    pc_vec = loadings[comp]
    idx = np.argsort(np.abs(pc_vec))[::-1][:top_n]

    # Handle cluster feature name correctly
    names = []
    for i in idx:
        if i < len(feature_names):
            names.append(feature_names[i])
        else:
            names.append("cluster_bin")

    plt.figure(figsize=(10, 6))
    colors = ["green" if v > 0 else "red" for v in pc_vec[idx]]
    plt.barh(names, pc_vec[idx], color=colors)
    plt.xlabel("Loading Value")
    plt.title(f"Top {top_n} Features → PC{comp+1}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"final_preprocessed_data/pca_visualizations/viz_pc{comp+1}_top_features.png", dpi=100)
    plt.close()

print("All visualizations saved!")


# ===========================
# STEP 12: SUMMARY REPORT
# ===========================
print("\n" + "="*60)
print("PREPROCESSING COMPLETE - SUMMARY REPORT")
print("="*60)

print("\n📊 DATASET SIZES:")
print(f"  Binary (train/test):      {len(yb_train):,} / {len(yb_test):,}")
print(f"  Multiclass (train/test):  {len(ym_train):,} / {len(ym_test):,}")
print(f"  Anomaly (train/test):     {len(ya_train):,} / {len(ya_test):,}")

print("\n💾 SAVED MODELS:")
print("  ✓ scaler.pkl")
print("  ✓ label_encoder.pkl")
print("  ✓ feature_names.pkl")
print("  ✓ kmeans_bin.pkl")
print("  ✓ kmeans_multi.pkl")
print("  ✓ pca_bin.pkl")
print("  ✓ pca_multi.pkl")

print("\n📁 SAVED CSV FILES:")
print("  ✓ binary_train.csv, binary_test.csv, binary_preprocessed.csv")
print("  ✓ multiclass_train.csv, multiclass_test.csv, multiclass_preprocessed.csv")
print("  ✓ anomaly_train.csv, anomaly_test.csv, anomaly_preprocessed.csv")

print("\n📈 VISUALIZATIONS:")
print("  ✓ Class distributions (3 files)")
print("  ✓ Correlation heatmap")
print("  ✓ PCA variance & scatter plots (4 files)")
print("  ✓ PC feature contributions (20 files)")

print("\n✅ All preprocessing complete!")
print("="*60)
