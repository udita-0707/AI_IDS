# AI-Based Intrusion Detection System (AI_IDS) - Codebase Summary

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture & Design](#architecture--design)
3. [Directory Structure](#directory-structure)
4. [Core Components](#core-components)
5. [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
6. [Machine Learning Models](#machine-learning-models)
7. [Inference Pipeline](#inference-pipeline)
8. [API Server](#api-server)
9. [Frontend Applications](#frontend-applications)
10. [Utilities](#utilities)
11. [Model Export & Optimization](#model-export--optimization)
12. [Workflow & Usage](#workflow--usage)
13. [Dependencies](#dependencies)
14. [Key Features](#key-features)

---

## Project Overview

**AI_IDS** is a comprehensive Network Intrusion Detection System that leverages machine learning and deep learning techniques to detect and classify network attacks. The system is built on the **CICIDS 2017 dataset** and implements three detection approaches:

1. **Binary Classification**: Distinguishes between BENIGN and ATTACK traffic
2. **Multiclass Classification**: Identifies specific attack types (14 attack categories)
3. **Anomaly Detection**: Detects unknown/zero-day attacks using unsupervised learning

The system can process PCAP files in real-time, extract network flows, and provide predictions through a RESTful API with two frontend interfaces.

---

## Architecture & Design

### High-Level Architecture

```
┌─────────────┐
│  PCAP/CSV   │
│    File     │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  Flow Extraction    │  (CICFlowMeter - if PCAP)
│  (utils/flow_extractor) │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Preprocessing       │  (Feature alignment, scaling, PCA)
│  (utils/preprocessing) │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│         Sequential Decision Logic        │
│  ┌──────────────┐  ┌──────────────┐    │
│  │ Binary XGBoost│  │Multiclass XGB│    │
│  │  (Gatekeeper) │  │  (Attack ID) │    │
│  └──────┬───────┘  └──────┬───────┘    │
│         │                 │             │
│         └────────┬────────┘             │
│                  ▼                       │
│         ┌─────────────────┐             │
│         │ Autoencoder      │             │
│         │ (Anomaly Detector)│             │
│         └─────────────────┘             │
└──────────────────┬──────────────────────┘
                   │
                   ▼
            ┌──────────────┐
            │  Predictions │
            │  CSV Output  │
            └──────────────┘
```

### Sequential Decision Logic

The inference pipeline uses a hierarchical decision-making approach:

1. **Stage 1 - Binary Classification (Gatekeeper)**: All flows are classified as BENIGN or ATTACK
2. **Stage 2 - Multiclass Classification**: Only high-confidence attacks (confidence ≥ 0.60) are classified into specific attack types
3. **Stage 3 - Anomaly Detection**: ALL flows are checked for anomalies using the autoencoder (runs independently)

**Final Label Assignment** (Priority Order):
- High-confidence attack (≥0.60) + has attack type → `ATTACK` + specific attack type
- High-confidence attack (≥0.60) + no attack type → `ANOMALY` + `UNKNOWN`
- Autoencoder detects anomaly (MSE > threshold) → `ANOMALY` + `UNKNOWN`
- Otherwise → `BENIGN` + `N/A`

**Key Thresholds**:
- `BIN_ATTACK_CONF_THRESHOLD = 0.60`: Minimum confidence for multiclass classification
- `ANOMALY_THRESHOLD_SENSITIVITY = 0.7`: Multiplier for saved threshold (70% = more sensitive)
- Autoencoder threshold: `saved_threshold * 0.7` (makes it more sensitive to detect anomalies)

---

## Directory Structure

```
AI_IDS/
├── api/
│   └── api_server.py              # FastAPI server for inference (port 8000)
├── binary_models/                  # Binary classification models (legacy/experimental)
│   ├── decision_tree_binary.py
│   ├── random_forest_binary.py
│   ├── svm_binary.py
│   └── xgBoost_binary.py
├── multiclass_models/              # Multiclass classification models (legacy/experimental)
│   ├── random_forest_multiclass.py
│   ├── tabnet_multiclass.py
│   └── xgBoost_multiclass.py
├── final_models/                   # Production-ready models
│   ├── final_xgboost_binary.py    # Binary XGBoost training
│   ├── final_xgboost_multi.py     # Multiclass XGBoost training
│   └── final_autoencoder_xgboost_anomaly.py  # Autoencoder training
├── inference/
│   └── live_predict.py            # Main inference pipeline
├── evaluate/
│   └── evaluate_models.py         # Model evaluation utilities
├── utils/
│   ├── __init__.py
│   ├── flow_extractor.py          # PCAP to flows conversion (CICFlowMeter)
│   └── preprocessing.py           # Feature preprocessing for inference
├── frontend/                       # React/TypeScript frontend (port 8080)
│   ├── src/
│   │   ├── App.tsx                # Main app component
│   │   ├── main.tsx               # Entry point
│   │   ├── pages/
│   │   │   ├── Index.tsx         # Main analysis page
│   │   │   └── NotFound.tsx      # 404 page
│   │   ├── components/
│   │   │   ├── FileUpload.tsx    # File upload component
│   │   │   ├── SummaryResults.tsx # Results display
│   │   │   ├── Header.tsx        # Navigation header
│   │   │   ├── LoadingAnimation.tsx
│   │   │   └── ui/               # shadcn-ui components
│   │   └── lib/
│   │       ├── api.ts            # API service (connects to backend)
│   │       └── utils.ts          # Utility functions
│   ├── package.json              # Node.js dependencies
│   ├── vite.config.ts            # Vite configuration
│   └── tailwind.config.ts        # Tailwind CSS config
├── live/                          # Runtime directories
│   ├── live_flows.csv            # Extracted flows (temporary)
│   ├── uploads/                  # Uploaded files (temporary)
│   └── predictions_*.csv         # Prediction outputs
├── final_preprocessed_data/       # Preprocessed datasets & models
│   ├── models/                   # Saved models & artifacts
│   │   ├── xgb_binary.pkl
│   │   ├── xgb_multiclass.pkl
│   │   ├── autoencoder_anomaly_only.h5
│   │   ├── scaler.pkl
│   │   ├── kmeans_bin.pkl
│   │   ├── kmeans_multi.pkl
│   │   ├── pca_bin.pkl
│   │   ├── pca_multi.pkl
│   │   ├── feature_names.pkl
│   │   ├── label_encoder.pkl
│   │   ├── reconstruction_threshold.pkl
│   │   ├── xgb_binary.onnx       # (optional) ONNX export
│   │   └── xgb_multiclass.onnx   # (optional) ONNX export
│   ├── csv/                      # Preprocessed datasets
│   │   ├── binary_preprocessed.csv
│   │   ├── binary_train.csv
│   │   ├── binary_test.csv
│   │   ├── multiclass_preprocessed.csv
│   │   ├── multiclass_train.csv
│   │   ├── multiclass_test.csv
│   │   ├── anomaly_train.csv
│   │   └── anomaly_test.csv
│   ├── plot_models/              # Training visualizations
│   │   ├── binary/
│   │   ├── multiclass_xgb/
│   │   └── anomaly/
│   └── stats_visualizations/     # Data statistics plots
├── preprocessed_binary_10k/      # Experimental binary models (10k samples)
│   └── plots_model/
├── preprocessed_multiclass_20k/  # Experimental multiclass models (20k samples)
│   └── plots_model/
├── binary_data_preprocess.py      # Binary preprocessing (legacy)
├── multi_data_preprocess.py       # Multiclass preprocessing (legacy)
├── final_data_preprocess.py       # Main preprocessing pipeline
├── export_to_onnx.py             # ONNX model conversion
├── quick_test.py                 # Quick testing script
├── project_report.html           # HTML project report
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## Core Components

### 1. Data Preprocessing (`final_data_preprocess.py`)

The main preprocessing pipeline that prepares the CICIDS 2017 dataset for all three model types.

**Key Steps**:
1. **Data Loading**: Loads all CSV files from `TrafficLabelling/` folder (~2.8M flows)
2. **Data Cleaning**: 
   - Removes non-numeric columns (Flow ID, IPs, timestamps, ports, protocol)
   - Handles infinite values and missing data
   - Removes duplicates
3. **Label Generation**:
   - Binary labels: 0 (BENIGN) or 1 (ATTACK)
   - Multiclass labels: Encoded attack types (14 classes)
   - Saves LabelEncoder for inference consistency
4. **Benign Reduction**: Downsamples benign traffic to 1.2x attack samples
5. **Feature Scaling**: StandardScaler normalization (Z-score)
6. **Class Balancing**:
   - Binary: Random oversampling for balance
   - Multiclass: Capping at 60,000 samples per class (no oversampling)
   - Anomaly: Normal-only dataset (80,000 samples max)
7. **Feature Engineering**:
   - KMeans clustering (6 clusters for binary, 10 for multiclass)
   - PCA dimensionality reduction (20 components, preserves 95%+ variance)
8. **Train/Test Split**: 75/25 stratified split (for final models)
9. **Artifact Saving**: Saves scalers, PCA models, KMeans models, feature names, label encoder

**Outputs**:
- Preprocessed CSV files for each task
- Saved preprocessing artifacts (scaler, PCA, KMeans, label encoder)
- Visualizations (class distributions, PCA plots, correlation heatmaps)

### 2. Binary Classification Models

#### Final XGBoost Binary (`final_models/final_xgboost_binary.py`)

**Model Configuration**:
- Algorithm: XGBoost Classifier
- Parameters:
  - `n_estimators`: 250
  - `max_depth`: 8
  - `learning_rate`: 0.08
  - `subsample`: 0.8
  - `colsample_bytree`: 0.8
  - `tree_method`: "hist"
  - `eval_metric`: "logloss"

**Training Process**:
1. Loads preprocessed binary dataset (PCA features)
2. 75/25 train/test split
3. Trains XGBoost model
4. Generates metrics: Accuracy, Precision, Recall, F1, ROC-AUC
5. Creates visualizations: Confusion matrix, ROC curve, PR curve, feature importance

**Performance**: ~99.9% accuracy, AUC ~1.00

**Output**: `final_preprocessed_data/models/xgb_binary.pkl`

### 3. Multiclass Classification Models

#### Final XGBoost Multiclass (`final_models/final_xgboost_multi.py`)

**Model Configuration**:
- Algorithm: XGBoost Classifier (multiclass)
- Parameters:
  - `objective`: "multi:softprob"
  - `n_estimators`: 300
  - `max_depth`: 10
  - `learning_rate`: 0.08
  - `subsample`: 0.8
  - `colsample_bytree`: 0.8
  - `eval_metric`: "mlogloss"
  - `tree_method`: "hist"

**Attack Types** (14 classes):
1. Bot
2. DDoS
3. DoS GoldenEye
4. DoS Hulk
5. DoS Slowhttptest
6. DoS slowloris
7. FTP-Patator
8. Heartbleed
9. Infiltration
10. PortScan
11. SSH-Patator
12. Web Attack - Brute Force
13. Web Attack - Sql Injection
14. Web Attack - XSS

**Performance**: ~97.5% accuracy

**Output**: `final_preprocessed_data/models/xgb_multiclass.pkl`

### 4. Anomaly Detection Models

#### Autoencoder Anomaly Detector (`final_models/final_autoencoder_xgboost_anomaly.py`)

**Architecture**:
- Input Layer: 20 (PCA dimensions)
- Encoder: Dense(32) → Dense(16) → Dense(10)
- Decoder: Dense(16) → Dense(32) → Dense(20)
- Activation: ReLU (hidden), Linear (output)
- Loss: Mean Squared Error (MSE)
- Optimizer: Adam (lr=0.001)
- Training: ONLY on normal (benign) traffic

**Training Process**:
1. Loads anomaly training data (normal-only, max 80k samples)
2. Trains autoencoder to reconstruct normal traffic patterns
3. Computes reconstruction error (MSE) for training data
4. Sets threshold: `mean + 3 * std` of reconstruction errors
5. Saves model and threshold

**Inference**:
- Runs on ALL flows (not just non-attacks)
- Computes reconstruction error for each flow
- Uses threshold: `saved_threshold * 0.7` (70% for sensitivity)
- Flows with MSE > threshold are flagged as anomalies

**Outputs**:
- `autoencoder_anomaly_only.h5`: Trained Keras model
- `reconstruction_threshold.pkl`: Detection threshold

---

## Data Preprocessing Pipeline

### Feature Engineering Pipeline

```
Raw Features (77 numeric features from CICFlowMeter)
    ↓
StandardScaler (Z-score normalization: mean=0, std=1)
    ↓
KMeans Clustering (adds cluster label as feature)
    - Binary: 6 clusters
    - Multiclass: 10 clusters
    ↓
PCA (20 principal components, preserves 95%+ variance)
    ↓
Model Input (20 features)
```

### Preprocessing Artifacts

All preprocessing artifacts are saved in `final_preprocessed_data/models/`:

1. **scaler.pkl**: StandardScaler fitted on training data
2. **feature_names.pkl**: List of feature names in correct order
3. **kmeans_bin.pkl**: KMeans model for binary classification (6 clusters)
4. **kmeans_multi.pkl**: KMeans model for multiclass (10 clusters)
5. **pca_bin.pkl**: PCA model for binary (20 components)
6. **pca_multi.pkl**: PCA model for multiclass (20 components)
7. **label_encoder.pkl**: LabelEncoder for multiclass attack types

---

## Machine Learning Models

### Model Comparison

| Model Type | Algorithm | Input Features | Output | Performance |
|------------|-----------|----------------|--------|-------------|
| Binary | XGBoost | 20 PCA components | BENIGN/ATTACK | ~99.9% accuracy, AUC 1.00 |
| Multiclass | XGBoost | 20 PCA components | 14 attack types | ~97.5% accuracy |
| Anomaly | Autoencoder | 20 PCA components | Normal/Anomaly | Detects novel attacks |

### Model Training Workflow

1. **Preprocess Data**: Run `final_data_preprocess.py`
2. **Train Binary Model**: Run `final_models/final_xgboost_binary.py`
3. **Train Multiclass Model**: Run `final_models/final_xgboost_multi.py`
4. **Train Anomaly Model**: Run `final_models/final_autoencoder_xgboost_anomaly.py`

All models are saved in `final_preprocessed_data/models/` for inference.

### Legacy/Experimental Models

The `binary_models/` and `multiclass_models/` directories contain experimental models trained on smaller datasets (10k binary, 20k multiclass) for model selection. These were used to compare algorithms before training on the full dataset.

---

## Inference Pipeline

### Main Inference Script (`inference/live_predict.py`)

The `run_pipeline()` function orchestrates the entire inference process:

#### Input Options:
- PCAP file (`.pcap` or `.pcapng`) - automatically extracts flows
- Pre-extracted flows CSV (`.csv`) - skips flow extraction

#### Processing Steps:

1. **Flow Extraction** (if PCAP):
   - Uses `utils/flow_extractor.py` to call CICFlowMeter
   - Extracts network flows and saves to `live/live_flows.csv`
   - Extracts 77+ flow features per connection

2. **Metadata Preservation**:
   - Preserves: Source IP, Destination IP, Source Port, Destination Port, Protocol, Timestamp
   - These are added back to the output CSV

3. **Preprocessing**:
   - Maps CICFlowMeter features to training format
   - Aligns features with training data (handles missing features)
   - Applies scaling, clustering, and PCA transformations
   - Uses saved artifacts from training

4. **Model Inference** (Sequential):
   - **Stage 1 - Binary Classification**: All flows → BENIGN/ATTACK
   - **Stage 2 - Multiclass Classification**: High-confidence attacks only (confidence ≥ 0.60)
   - **Stage 3 - Anomaly Detection**: ALL flows checked (runs independently)

5. **Decision Logic**:
   ```python
   if high_confidence_attack (≥0.60) and has_attack_type:
       label = "ATTACK"
       attack_type = multiclass_prediction
   elif high_confidence_attack (≥0.60) and no_attack_type:
       label = "ANOMALY"
       attack_type = "UNKNOWN"
   elif autoencoder_detects_anomaly:
       label = "ANOMALY"
       attack_type = "UNKNOWN"
   else:
       label = "BENIGN"
       attack_type = "N/A"
   ```

6. **Output Generation**:
   - CSV file with columns:
     - Flow_ID, Source_IP, Destination_IP, Source_Port, Destination_Port
     - Protocol, Timestamp, Label, Attack_Type
     - Binary_Label, Binary_Confidence
     - Anomaly_Detected, Anomaly_Confidence, Reconstruction_MSE

#### Usage:
```bash
# From PCAP file
python inference/live_predict.py --pcap input.pcap --out output.csv

# From flows CSV
python inference/live_predict.py --csv flows.csv --out output.csv

# Debug mode (shows detailed MSE statistics)
python inference/live_predict.py --pcap input.pcap --out output.csv --debug
```

---

## API Server

### FastAPI Server (`api/api_server.py`)

A production-ready REST API for real-time network traffic analysis.

#### Endpoints:

1. **GET `/`**: Root/Health check
   - Returns API status and available endpoints

2. **GET `/health`**: Detailed health check
   - Returns model loading status (binary, multiclass, anomaly)
   - Timestamp

3. **POST `/predict`**: Main inference endpoint
   - **Input**: PCAP/PCAPNG or CSV file (multipart/form-data)
   - **Output**: JSON response with:
     - Status, file_type, filename, file_size_bytes
     - Total flows processed
     - Summary (label counts: BENIGN, ATTACK, ANOMALY)
     - Attack type breakdown (if attacks detected)
     - Download CSV link
     - All flows data (complete dataset, not just preview)

4. **GET `/download/{filename}`**: Download prediction CSV
   - Returns the generated CSV file
   - Filename format: `predictions_{timestamp}.csv`

5. **POST `/predict_pcap`**: (Deprecated, use `/predict`)
   - Backward compatibility endpoint

#### Features:
- CORS enabled for cross-origin requests (all origins)
- Automatic file validation (`.pcap`, `.pcapng`, `.csv`)
- Temporary file handling with cleanup
- Error handling with detailed messages
- Automatic flow extraction from PCAP files
- Supports both PCAP and CSV inputs

#### Usage:
```bash
# Start server
python api/api_server.py

# Server runs on http://0.0.0.0:8000
# API docs available at http://localhost:8000/docs
```

#### Example Request:
```bash
# Upload PCAP file
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@network_traffic.pcap"

# Upload CSV file
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@flows.csv"
```

---

## Frontend Applications

### React/TypeScript Frontend (`frontend/`)

A modern web interface built with React, TypeScript, Vite, and shadcn-ui.

#### Technology Stack:
- **Framework**: React 18.3.1
- **Language**: TypeScript 5.8.3
- **Build Tool**: Vite 5.4.19
- **UI Library**: shadcn-ui (Radix UI components)
- **Styling**: Tailwind CSS 3.4.17
- **Routing**: React Router DOM 6.30.1
- **Icons**: Lucide React

#### Key Components:

1. **Index.tsx** (`src/pages/Index.tsx`):
   - Main analysis page
   - File upload interface
   - Results display
   - API health checking

2. **FileUpload.tsx** (`src/components/FileUpload.tsx`):
   - Drag & drop file upload
   - File validation (.pcap, .csv)
   - Visual feedback

3. **SummaryResults.tsx** (`src/components/SummaryResults.tsx`):
   - Displays summary statistics (BENIGN, ATTACK, ANOMALY counts)
   - Attack type breakdown
   - Processing information
   - CSV download button

4. **api.ts** (`src/lib/api.ts`):
   - API service layer
   - Functions: `checkApiHealth()`, `predictFile()`, `getDownloadUrl()`
   - Configurable API base URL (default: http://localhost:8000)

#### Features:
- Modern, responsive UI
- Real-time API integration
- File upload with drag & drop
- Summary statistics visualization
- Attack type breakdown
- CSV download functionality
- Error handling and user feedback

#### Usage:
```bash
# Install dependencies
cd frontend
npm install

# Start development server
npm run dev

# Frontend runs on http://localhost:8080
# Make sure backend API is running on port 8000
```

#### Configuration:
- API URL can be configured via environment variable: `VITE_API_BASE_URL`
- Default: `http://localhost:8000`
- Edit `src/lib/api.ts` to change default URL

---

## Utilities

### 1. Flow Extractor (`utils/flow_extractor.py`)

**Purpose**: Converts PCAP files to network flow features using CICFlowMeter.

**Function**: `extract_flows_from_pcap(input_pcap, output_csv)`

**Process**:
- Uses CICFlowMeter Python library (pure Python)
- Extracts 77+ flow features per network connection using Scapy
- Outputs CSV with flow statistics

**Features Extracted**:
- Flow duration, packet counts, byte counts
- Packet length statistics (mean, std, min, max)
- Inter-arrival times (IAT)
- TCP flags (SYN, ACK, FIN, etc.)
- Forward/backward packet ratios
- Subflow statistics
- And more...

**Requirements**: Scapy (for packet processing)

### 2. Preprocessing Utilities (`utils/preprocessing.py`)

**Key Functions**:

1. **`load_preprocessing_artifacts(models_dir)`**:
   - Loads scaler, KMeans models, PCA models, feature names, label encoder
   - Returns dictionary of artifacts
   - Used during inference to apply same transformations as training

2. **`preprocess_for_models(df_raw, artifacts)`**:
   - Maps CICFlowMeter features to training format
   - Aligns features with training data (handles missing/extra features)
   - Applies scaling, clustering, PCA
   - Returns preprocessed arrays for binary, multiclass, and anomaly models
   - Handles NaN/inf values (fills with zeros)

**Feature Mapping**:
- Automatically detects CICFlowMeter format
- Maps feature names (e.g., `flow_duration` → `Flow_Duration`)
- Handles missing features (fills with zeros)
- Handles NaN/inf values (fills with zeros)

---

## Model Export & Optimization

### ONNX Export (`export_to_onnx.py`)

Converts trained XGBoost models to ONNX format for optimized inference and cross-platform deployment.

**Process**:
1. Loads trained XGBoost models (binary and multiclass)
2. Converts to ONNX format using `onnxmltools`
3. Saves as `.onnx` files

**Outputs**:
- `final_preprocessed_data/models/xgb_binary.onnx`
- `final_preprocessed_data/models/xgb_multiclass.onnx`

**Usage**:
```bash
python export_to_onnx.py
```

**Benefits**:
- Faster inference (optimized runtime)
- Cross-platform compatibility
- Smaller model size
- Integration with ONNX Runtime

**Note**: The current inference pipeline uses pickle files, but ONNX support demonstrates optimization capabilities.

---

## Workflow & Usage

### Complete Workflow

#### 1. Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies (optional)
cd frontend
npm install
cd ..

# Ensure CICIDS 2017 dataset is in TrafficLabelling/ folder
```

#### 2. Data Preprocessing
```bash
# Run main preprocessing pipeline
python final_data_preprocess.py

# This creates:
# - final_preprocessed_data/csv/*.csv
# - final_preprocessed_data/models/*.pkl (artifacts)
# - Visualizations
```

#### 3. Model Training
```bash
# Train binary classifier
python final_models/final_xgboost_binary.py

# Train multiclass classifier
python final_models/final_xgboost_multi.py

# Train anomaly detector
python final_models/final_autoencoder_xgboost_anomaly.py
```

#### 4. Model Evaluation (Optional)
```bash
# Evaluate models on test sets
python evaluate/evaluate_models.py \
    --binary_test final_preprocessed_data/csv/binary_test.csv \
    --multi_test final_preprocessed_data/csv/multiclass_test.csv \
    --anomaly_test final_preprocessed_data/csv/anomaly_test.csv
```

#### 5. Model Export (Optional)
```bash
# Export models to ONNX format
python export_to_onnx.py
```

#### 6. Inference (Command Line)
```bash
# Process PCAP file
python inference/live_predict.py --pcap network.pcap --out predictions.csv

# Process flows CSV
python inference/live_predict.py --csv flows.csv --out predictions.csv

# Debug mode
python inference/live_predict.py --pcap network.pcap --out predictions.csv --debug
```

#### 7. Inference (API Server)
```bash
# Start API server
python api/api_server.py

# Server runs on http://localhost:8000
# API docs: http://localhost:8000/docs
```

#### 8. Frontend (Optional)
```bash
# Start frontend development server
cd frontend
npm run dev

# Frontend runs on http://localhost:8080
# Make sure backend API is running on port 8000
```

---

## Dependencies

### Core Libraries
- **numpy** (>=1.26.0, <2.0.0): Numerical computations
- **pandas** (>=2.2.0): Data manipulation
- **scikit-learn** (>=1.5.0): Machine learning utilities

### Machine Learning
- **xgboost** (==2.0.3): Gradient boosting for binary/multiclass
- **tensorflow** (>=2.13.0): Deep learning (autoencoder)
- **imbalanced-learn** (==0.12.0): Class balancing

### Deep Learning
- **pytorch-tabnet**: TabNet model (legacy multiclass experiments)

### Network Analysis
- **scapy**: Packet manipulation (required for CICFlowMeter)
- **pyshark**: Wireshark integration (optional)
- **cicflowmeter** (GitHub): Python-based flow feature extraction (no Java required)

### API & Server
- **fastapi** (>=0.104.0): REST API framework
- **uvicorn** (>=0.24.0): ASGI server
- **python-multipart** (>=0.0.6): File upload support

### Visualization
- **matplotlib** (>=3.8.0): Plotting
- **seaborn** (>=0.13.0): Statistical visualizations

### Model Conversion
- **onnx** (>=1.15.0): Model interoperability
- **onnxruntime** (>=1.16.0): ONNX inference
- **onnxmltools** (>=1.11.0): Model conversion

### Frontend (Node.js)
- **react** (^18.3.1): UI framework
- **typescript** (^5.8.3): Type safety
- **vite** (^5.4.19): Build tool
- **tailwindcss** (^3.4.17): Styling
- **@radix-ui/***: UI component library
- See `frontend/package.json` for complete list

---

## Key Features

### 1. Multi-Model Ensemble
- Combines supervised (XGBoost) and unsupervised (Autoencoder) approaches
- Sequential decision logic for optimal performance
- Confidence thresholds for reliable predictions
- Independent anomaly detection on all flows

### 2. Real-Time Processing
- PCAP file processing with automatic flow extraction
- Fast inference pipeline optimized for production
- RESTful API for integration with other systems
- Supports both PCAP and CSV inputs

### 3. Comprehensive Attack Detection
- **Binary**: Quick BENIGN vs ATTACK classification (~99.9% accuracy)
- **Multiclass**: Identifies 14 specific attack types (~97.5% accuracy)
- **Anomaly**: Detects unknown/zero-day attacks (unsupervised)

### 4. Robust Preprocessing
- Automatic feature alignment
- Handles missing/invalid data gracefully
- Consistent preprocessing for training and inference
- Preserves metadata (IPs, ports, timestamps)

### 5. Production-Ready
- FastAPI server with error handling
- Model persistence (pickle, H5)
- Comprehensive logging and monitoring
- CORS enabled for frontend integration
- Modern React frontend interface

### 6. Extensible Architecture
- Modular design for easy model swapping
- Utility functions for custom preprocessing
- Evaluation tools for model assessment
- ONNX export for optimization

### 7. Dual Frontend Support
- React/TypeScript frontend (modern UI, port 8080)
- API-first design (can integrate with any frontend)
- Real-time API health checking
- File upload with drag & drop

---

## Attack Types Detected

The multiclass classifier can identify the following attack types:

1. **Bot**: Botnet traffic
2. **DDoS**: Distributed Denial of Service
3. **DoS GoldenEye**: DoS attack variant
4. **DoS Hulk**: DoS attack variant
5. **DoS Slowhttptest**: Slow HTTP test attack
6. **DoS slowloris**: Slowloris attack
7. **FTP-Patator**: FTP brute force
8. **Heartbleed**: Heartbleed vulnerability exploit
9. **Infiltration**: Network infiltration
10. **PortScan**: Port scanning activity
11. **SSH-Patator**: SSH brute force
12. **Web Attack - Brute Force**: Web application brute force
13. **Web Attack - Sql Injection**: SQL injection attempt
14. **Web Attack - XSS**: Cross-site scripting attack

---

## Performance Characteristics

### Binary Classification
- **Accuracy**: ~99.9%
- **Precision**: Very high (minimal false positives)
- **Recall**: High (catches most attacks)
- **AUC**: ~1.00
- **Speed**: Fast inference (<1ms per flow)

### Multiclass Classification
- **Accuracy**: ~97.5%
- **Macro F1**: Balanced performance across classes
- **Speed**: Moderate (only runs on high-confidence attacks, ≥0.60)
- **Classes**: 14 attack types

### Anomaly Detection
- **Sensitivity**: Configurable via threshold (70% of saved threshold)
- **Novel Attack Detection**: Can detect attacks not seen during training
- **Speed**: Moderate (runs on all flows)
- **Threshold**: `saved_threshold * 0.7` (more sensitive)

---

## File Naming Conventions

- **Preprocessing scripts**: `*_data_preprocess.py`
- **Model training**: `final_models/final_*.py`
- **Legacy models**: `binary_models/*.py`, `multiclass_models/*.py`
- **Inference**: `inference/live_predict.py`
- **API**: `api/api_server.py`
- **Utilities**: `utils/*.py`
- **Evaluation**: `evaluate/*.py`
- **Export**: `export_to_onnx.py`

---

## Notes & Considerations

1. **Dataset Requirements**: CICIDS 2017 dataset must be placed in `TrafficLabelling/` folder
2. **Model Loading**: Models are loaded from `final_preprocessed_data/models/` during inference
3. **Temporary Files**: Flow extraction creates temporary CSV files in `live/` directory
4. **Memory Usage**: Large PCAP files may require significant memory for processing
5. **CICFlowMeter**: Python library (no Java required) - uses Scapy for packet processing
6. **TensorFlow**: Autoencoder requires TensorFlow (can be disabled if unavailable)
7. **API Port**: Backend API runs on port 8000 by default
8. **Frontend Port**: React frontend runs on port 8080 by default
9. **Anomaly Sensitivity**: Threshold sensitivity can be adjusted in `inference/live_predict.py` (ANOMALY_THRESHOLD_SENSITIVITY)

---

## Future Enhancements

Potential improvements and extensions:

1. **Online Learning**: Incremental model updates from new data
2. **Deep Learning**: LSTM/CNN for temporal pattern detection
3. **Feature Selection**: Automated feature importance analysis
4. **Model Ensembling**: Combine multiple models for better accuracy
5. **Real-Time Streaming**: Process network traffic in real-time
6. **Enhanced Dashboard**: More visualization options in frontend
7. **Alerting System**: Integration with SIEM systems
8. **Model Versioning**: Track model versions and performance over time
9. **Batch Processing**: Process multiple files simultaneously
10. **Model Monitoring**: Track model performance over time

---

## Conclusion

The AI_IDS codebase is a comprehensive, production-ready intrusion detection system that combines multiple machine learning approaches for robust network security. It provides both command-line and API interfaces for flexible deployment and integration, with a modern React frontend for user interaction.

The system's strength lies in its:
- **Multi-layered detection**: Binary → Multiclass → Anomaly
- **Robust preprocessing**: Handles real-world data variations
- **Production-ready API**: Easy integration with existing systems
- **Comprehensive coverage**: Detects known attacks and anomalies
- **Modern frontend**: User-friendly interface for analysis
- **Flexible input**: Supports both PCAP and CSV files

For questions or contributions, refer to the main README.md file.

---

**Last Updated**: 6-Dec-2025
**Version**: 2.0 (API-enabled inference system with React frontend)
