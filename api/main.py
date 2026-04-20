"""
API FastAPI — détection de déchets via 8 modèles MLflow.
"""
import torch
import io
import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
DB_PATH = os.environ.get("APP_DB_PATH", "/data/app_detections.db")
LOG_PATH = os.environ.get("LOG_PATH", "/app/logs/predictions.jsonl")
MAX_IMAGE_BYTES = 10 * 1024 * 1024
ALLOWED_MIME = {"image/jpeg", "image/png", "image/jpg"}

MODEL_SHORT_NAMES = [
    "yolov8", "yolo26", "rtdetr", "rtdetrv2",
    "rfdetr", "dfine", "deim-dfine", "fusion-model",
]

# -----------------------------------------------------------------------------
# Metrics Prometheus
# -----------------------------------------------------------------------------
predictions_total = Counter("ml_predictions_total", "Total predictions")
predictions_by_model = Counter(
    "ml_predictions_by_model_total", "Predictions by model", ["model"]
)
validation_errors = Counter("ml_validation_errors_total", "Validation errors")
inference_latency = Histogram(
    "ml_inference_latency_seconds", "Inference latency in seconds"
)

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = FastAPI(title="Waste Detection API", version="1.0.0")
MODELS: dict = {}


def init_db():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS app_detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            confiance REAL NOT NULL,
            model_name TEXT NOT NULL,
            source TEXT NOT NULL CHECK (source IN ('manual','drone_patrol')),
            drone_id TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_prediction(entry: dict):
    Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


@app.on_event("startup")
def startup():
    init_db()
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.MlflowClient()

    for short in MODEL_SHORT_NAMES:
        registry_name = f"waste-detector-{short}"
        uri = f"models:/{registry_name}/Production"
        try:
            model = mlflow.pyfunc.load_model(uri)
            versions = client.get_latest_versions(registry_name, stages=["Production"])
            version = versions[0] if versions else None
            registered_model = client.get_registered_model(registry_name)
            MODELS[short] = {
                "model": model,
                "version": version.version if version else "?",
                "registered_at": datetime.fromtimestamp(
                    registered_model.creation_timestamp / 1000, tz=timezone.utc
                ).isoformat(),
            }
            print(f"✓ Loaded {short} (v{MODELS[short]['version']})")
        except Exception as e:
            print(f"✗ Failed to load {short}: {e}")


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": len(MODELS)}


@app.get("/models")
def list_models():
    return [
        {
            "name": name,
            "version": info["version"],
            "registered_at": info["registered_at"],
        }
        for name, info in MODELS.items()
    ]


@app.post("/predict")
async def predict(
    file: UploadFile,
    latitude: float = Form(...),
    longitude: float = Form(...),
    model_name: str = Form(...),
    source: str = Form("manual"),
    drone_id: str | None = Form(None),
):
    # --- Validation modèle ---
    if model_name not in MODELS:
        validation_errors.inc()
        raise HTTPException(
            422,
            f"Modèle inconnu '{model_name}'. Modèles valides : {list(MODELS.keys())}",
        )

    # --- Validation fichier ---
    if file.content_type not in ALLOWED_MIME:
        validation_errors.inc()
        raise HTTPException(
            422,
            f"Le fichier doit être JPEG ou PNG (reçu : {file.content_type})",
        )

    data = await file.read()
    if len(data) > MAX_IMAGE_BYTES:
        validation_errors.inc()
        raise HTTPException(422, f"Image trop grande (max 10 Mo, reçu {len(data)/1e6:.1f} Mo)")

    # --- Validation GPS ---
    if not (-90 <= latitude <= 90):
        validation_errors.inc()
        raise HTTPException(422, f"Latitude invalide : {latitude} (doit être entre -90 et 90)")
    if not (-180 <= longitude <= 180):
        validation_errors.inc()
        raise HTTPException(422, f"Longitude invalide : {longitude} (doit être entre -180 et 180)")

    # --- Décode image ---
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        arr = np.array(img)
    except Exception as e:
        validation_errors.inc()
        raise HTTPException(422, f"Impossible de décoder l'image : {e}")

    # --- Validation source ---
    if source not in ("manual", "drone_patrol"):
        validation_errors.inc()
        raise HTTPException(422, f"Source invalide : {source}")

    # --- Inférence ---
    t0 = time.perf_counter()
    with inference_latency.time():
        result = MODELS[model_name]["model"].predict([arr])
    latency_ms = (time.perf_counter() - t0) * 1000

    # Le wrapper retourne {rubbish, confiance} (dict) ou [{...}] (liste)
    if isinstance(result, list):
        result = result[0]
    rubbish = bool(result["rubbish"])
    confiance = float(result["confiance"])
    ts = datetime.now(timezone.utc).isoformat()

    # --- Persiste ---
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO app_detections
           (timestamp, latitude, longitude, confiance, model_name, source, drone_id)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (ts, latitude, longitude, confiance, model_name, source, drone_id),
    )
    conn.commit()
    conn.close()

    # --- Metrics + log JSON ---
    predictions_total.inc()
    predictions_by_model.labels(model=model_name).inc()
    log_prediction({
        "timestamp": ts,
        "source": source,
        "latitude": latitude,
        "longitude": longitude,
        "confiance": confiance,
        "model_name": model_name,
        "latence_ms": round(latency_ms, 2),
    })

    return {
        "rubbish": rubbish,
        "confiance": confiance,
        "model_used": model_name,
        "timestamp": ts,
    }


@app.get("/history")
def history():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM app_detections ORDER BY timestamp DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)