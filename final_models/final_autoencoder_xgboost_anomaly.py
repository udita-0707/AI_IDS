"""
AUTOENCODER-ONLY ANOMALY DETECTION MODEL
Trains ONLY on anomaly_train.csv (normal traffic).
Outputs:
  ✓ autoencoder_anomaly_only.h5
  ✓ reconstruction_threshold.pkl
  ✓ training visualizations (optional)
"""

import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam


# ================================
# PATHS
# ================================
BASE_DIR = "final_preprocessed_data"
TRAIN_CSV = f"{BASE_DIR}/csv/anomaly_train.csv"
MODEL_DIR = f"{BASE_DIR}/models"
PLOT_DIR = f"{BASE_DIR}/plot_models/anomaly"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

print(f"\nModels will be saved to: {MODEL_DIR}/")
print(f"Plots will be saved to : {PLOT_DIR}/")


# =====================================================
# AUTOENCODER-ONLY CLASS
# =====================================================
class AutoencoderAnomalyDetector:
    def __init__(self):
        self.autoencoder = None
        self.reconstruction_threshold = None
        self.history = None

    # -------------------------------------------------
    # 1. Load training data
    # -------------------------------------------------
    def load_data(self):
        print("\nLoading training data (normal-only)...")
        df = pd.read_csv(TRAIN_CSV)

        self.X_train = df.drop(columns=["Label"]).values
        self.y_train = df["Label"].values

        print(f"✓ Loaded: {self.X_train.shape[0]} normal samples")
        print(f"✓ Feature dimension: {self.X_train.shape[1]}")

    # -------------------------------------------------
    # 2. Build autoencoder
    # -------------------------------------------------
    def build_autoencoder(self, input_dim, encoding_dim=10):
        input_layer = layers.Input(shape=(input_dim,))

        encoded = layers.Dense(32, activation="relu")(input_layer)
        encoded = layers.Dense(16, activation="relu")(encoded)
        encoded = layers.Dense(encoding_dim, activation="relu")(encoded)

        decoded = layers.Dense(16, activation="relu")(encoded)
        decoded = layers.Dense(32, activation="relu")(decoded)
        decoded = layers.Dense(input_dim, activation="linear")(decoded)

        model = keras.Model(input_layer, decoded)
        model.compile(optimizer=Adam(0.001), loss="mse")

        return model

    # -------------------------------------------------
    # 3. Train autoencoder
    # -------------------------------------------------
    def train(self, epochs=50, batch_size=32):
        print("\nTraining Autoencoder (normal traffic only)...")
        input_dim = self.X_train.shape[1]

        self.autoencoder = self.build_autoencoder(input_dim)
        self.autoencoder.summary()

        early_stop = callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
        )

        start = time.time()
        self.history = self.autoencoder.fit(
            self.X_train,
            self.X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=1,
        )
        print(f"\n✓ Training completed in {time.time() - start:.2f}s")

        # Compute reconstruction threshold
        X_rec = self.autoencoder.predict(self.X_train, verbose=0)
        train_mse = np.mean((self.X_train - X_rec) ** 2, axis=1)

        self.reconstruction_threshold = train_mse.mean() + 3 * train_mse.std()

        print(f"\nReconstruction Threshold: {self.reconstruction_threshold:.6f}")

    # -------------------------------------------------
    # 4. Save model + threshold
    # -------------------------------------------------
    def save(self):
        model_path = f"{MODEL_DIR}/autoencoder_anomaly_only.h5"
        threshold_path = f"{MODEL_DIR}/reconstruction_threshold.pkl"

        self.autoencoder.save(model_path)
        with open(threshold_path, "wb") as f:
            pickle.dump(self.reconstruction_threshold, f)

        print(f"\n✓ Saved Autoencoder: {model_path}")
        print(f"✓ Saved Threshold  : {threshold_path}")

    # -------------------------------------------------
    # 5. Optional: Create training plots
    # -------------------------------------------------
    def create_plots(self):
        print("\nGenerating training visualizations...")

        # 1. Loss curves
        plt.figure(figsize=(8, 5))
        plt.plot(self.history.history["loss"], label="Training Loss")
        plt.plot(self.history.history["val_loss"], label="Validation Loss")
        plt.title("Autoencoder Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

        path1 = f"{PLOT_DIR}/autoencoder_loss_curve.png"
        plt.savefig(path1, dpi=300, bbox_inches="tight")
        plt.close()

        # 2. Reconstruction error histogram
        X_rec = self.autoencoder.predict(self.X_train, verbose=0)
        train_mse = np.mean((self.X_train - X_rec) ** 2, axis=1)

        plt.figure(figsize=(8, 5))
        plt.hist(train_mse, bins=50, alpha=0.7)
        plt.axvline(self.reconstruction_threshold, color="red", linestyle="--", linewidth=2)
        plt.title("Reconstruction Error Distribution (Normal Training Data)")
        plt.xlabel("MSE")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)

        path2 = f"{PLOT_DIR}/reconstruction_error_histogram.png"
        plt.savefig(path2, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✓ Saved plot: {path1}")
        print(f"✓ Saved plot: {path2}")

    # -------------------------------------------------
    # RUN EVERYTHING
    # -------------------------------------------------
    def run(self, epochs=50, create_plots=True):
        self.load_data()
        self.train(epochs=epochs)
        self.save()
        if create_plots:
            self.create_plots()

        print("\n==========================================")
        print("AUTOENCODER-ONLY ANOMALY MODEL TRAINED!")
        print("==========================================\n")


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    detector = AutoencoderAnomalyDetector()
    detector.run(epochs=50, create_plots=True)
