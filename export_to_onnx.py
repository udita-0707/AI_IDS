import pickle
import pandas as pd
from pathlib import Path
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

MODELS_DIR = Path("final_preprocessed_data/models")
TRAIN_CSV = Path("final_preprocessed_data/csv/binary_preprocessed.csv")

print("\n=== EXPORTING XGBOOST MODELS TO ONNX ===")

df = pd.read_csv(TRAIN_CSV)

label_cols = ["Binary_Label"]
features = [c for c in df.columns if c not in label_cols]

n_features = len(features)
initial_type = [('input', FloatTensorType([None, n_features]))]

# Load models
with open(MODELS_DIR / "xgb_binary.pkl", "rb") as f:
    xgb_bin = pickle.load(f)

with open(MODELS_DIR / "xgb_multiclass.pkl", "rb") as f:
    xgb_multi = pickle.load(f)

# Convert
binary_onnx = convert_xgboost(xgb_bin, initial_types=initial_type)
multi_onnx = convert_xgboost(xgb_multi, initial_types=initial_type)

# Save
(MODELS_DIR / "xgb_binary.onnx").write_bytes(binary_onnx.SerializeToString())
(MODELS_DIR / "xgb_multiclass.onnx").write_bytes(multi_onnx.SerializeToString())

print("\nSUCCESS — Models exported to ONNX!")