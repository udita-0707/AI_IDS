#!/usr/bin/env python3
"""
FastAPI server that accepts PCAP or CSV upload and returns JSON table + CSV file.
POST /predict  (form-data: file)
Supports both .pcap/.pcapng files and .csv flow files.
"""

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import tempfile
import os
import sys
import time

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from inference.live_predict import run_pipeline

app = FastAPI(
    title="AI_IDS Inference API",
    description="Network Intrusion Detection System with Binary, Multiclass, and Anomaly Detection",
    version="2.0"
)

# ===== FIX FOR POSTMAN/HOPSCOTCH: ADD CORS =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

UPLOAD_DIR = Path("live/uploads")
OUTPUT_DIR = Path("live")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "AI_IDS Inference API",
        "endpoints": {
            "predict": "/predict (POST) - accepts PCAP or CSV files",
            "docs": "/docs",
            "health": "/"
        }
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "models": {
            "binary": "loaded",
            "multiclass": "loaded",
            "anomaly": "loaded"
        }
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload a PCAP or CSV file and get network flow predictions
    
    Args:
        file: PCAP/PCAPNG file OR CSV file with network flows
        
    Returns:
        JSON with predictions and download link
        
    Supported formats:
        - PCAP files: .pcap, .pcapng (will extract flows automatically)
        - CSV files: .csv (should contain network flow features)
    """
    print(f"\n[API] Received file: {file.filename}")
    print(f"[API] Content-Type: {file.content_type}")
    
    # Validate file extension and determine file type
    suffix = Path(file.filename).suffix.lower()
    is_pcap_file = suffix in [".pcap", ".pcapng"]
    is_csv_file = suffix == ".csv"
    
    if not (is_pcap_file or is_csv_file):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type '{suffix}'. Upload a .pcap, .pcapng, or .csv file"
        )

    # Create temp directory for upload
    tmp_dir = tempfile.mkdtemp(prefix="aiids_")
    tmp_file = Path(tmp_dir) / f"upload{suffix}"
    
    file_type = "PCAP" if is_pcap_file else "CSV"
    print(f"[API] Detected file type: {file_type}")
    print(f"[API] Saving to temporary location: {tmp_file}")
    
    try:
        # Save uploaded file
        with open(tmp_file, "wb") as out_f:
            shutil.copyfileobj(file.file, out_f)
        
        file_size = tmp_file.stat().st_size
        print(f"[API] File saved: {file_size} bytes")
        
        # Run inference pipeline
        # If PCAP: is_pcap=True will trigger flow extraction
        # If CSV: is_pcap=False will skip flow extraction and go directly to preprocessing
        out_csv = OUTPUT_DIR / f"predictions_{int(time.time())}.csv"
        print(f"[API] Running inference pipeline (is_pcap={is_pcap_file})...")
        
        df_out = run_pipeline(str(tmp_file), str(out_csv), is_pcap=is_pcap_file)
        
        print(f"[API] Pipeline complete. Generated {len(df_out)} predictions")
        
        # Prepare response
        label_counts = df_out['Label'].value_counts().to_dict()
        attack_breakdown = {}
        
        if 'Attack_Type' in df_out.columns:
            attack_df = df_out[df_out['Label'] == 'ATTACK']
            if len(attack_df) > 0:
                attack_breakdown = attack_df['Attack_Type'].value_counts().to_dict()
        
        # Return all flows, not just preview
        # For very large datasets, this might be memory intensive, but user requested all flows
        all_flows = df_out.to_dict(orient="records")
        
        response = {
            "status": "success",
            "file_type": file_type,
            "filename": file.filename,
            "file_size_bytes": file_size,
            "total_flows": len(df_out),
            "summary": label_counts,
            "attack_types": attack_breakdown,
            "download_csv": str(out_csv),
            "data_preview": all_flows,  # Changed from head(50) to all flows
            "all_flows": all_flows  # Explicitly include all flows
        }
        
        print(f"[API] Returning response with {len(all_flows)} flows (all flows included)")
        return JSONResponse(content=response)
        
    except Exception as e:
        print(f"[API] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
        
    finally:
        # Cleanup uploaded file
        try:
            if tmp_file.exists():
                os.remove(tmp_file)
            if Path(tmp_dir).exists():
                os.rmdir(tmp_dir)
            print(f"[API] Cleanup complete")
        except Exception as cleanup_err:
            print(f"[API] Cleanup warning: {cleanup_err}")


# Keep backward compatibility with old endpoint name
@app.post("/predict_pcap")
async def predict_pcap(file: UploadFile = File(...)):
    """
    [DEPRECATED] Use /predict instead. This endpoint is kept for backward compatibility.
    Upload a PCAP file and get network flow predictions
    """
    return await predict(file)


@app.get("/download/{filename}")
async def download_csv(filename: str):
    """
    Download a prediction CSV file
    
    Args:
        filename: Name of the CSV file (e.g., predictions_1234567890.csv)
    """
    file_path = OUTPUT_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        media_type="text/csv",
        filename=filename
    )


if __name__ == "__main__":
    print("=" * 70)
    print("AI_IDS INFERENCE API SERVER")
    print("=" * 70)
    print("Starting server on http://0.0.0.0:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("=" * 70)
    
    uvicorn.run(
        "api.api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )