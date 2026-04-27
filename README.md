# 🗑️ Waste Detection MLOps

![CI]https://github.com/GASTLINASSIM/waste-detection-mlops-eceparis.git

A production-grade MLOps platform for real-time waste detection using drone patrol imagery. The system exposes a multi-model inference API, an automated ETL pipeline, a Streamlit dashboard, and a full observability stack — all orchestrated via Docker Compose.

**Authors**: Nassim Gastli · Bilel Sahnoun  
**Program**: MSc 2 Data Management & AI — ECE Paris

---

## Table of Contents

- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Models](#models)
- [ETL Pipeline](#etl-pipeline)
- [Dashboard](#dashboard)
- [Observability](#observability)
- [CI/CD](#cicd)
- [Testing](#testing)
- [Project Structure](#project-structure)

---

## Architecture

The platform is composed of 7 services running in Docker:

| Service | Role | Port |
|---|---|---|
| `api` | FastAPI inference & history endpoints | 8000 |
| `app` | Streamlit dashboard | 8501 |
| `airflow` | DAG orchestration (ETL pipeline) | 8080 |
| `mlflow` | Model registry | 5000 |
| `prometheus` | Metrics scraping | 9090 |
| `grafana` | Monitoring dashboard | 3000 |
| `alertmanager` | Alerting | 9093 |

---

## Prerequisites

- Docker & Docker Compose
- Python 3.9+
- `curl`, `sqlite3`, `pytest`

---

## Getting Started

**1. Clone the repository**

```bash
git clone https://github.com/GASTLINASSIM/waste-detection-mlops-eceparis.git
cd waste-detection-mlops-eceparis
```

**2. Generate the drone patrol database**

```bash
python generate_patrol_db.py
```

Expected output:
```
✓ Mission simulated — drone_patrol.db updated
  XX new detections inserted
```

**3. Start the full stack**

```bash
docker compose up -d
docker compose ps
```

All 7 services should be in `running` state:

```
NAME          STATUS    PORTS
api           running   0.0.0.0:8000->8000/tcp
app           running   0.0.0.0:8501->8501/tcp
airflow       running   0.0.0.0:8080->8080/tcp
mlflow        running   0.0.0.0:5000->5000/tcp
prometheus    running   0.0.0.0:9090->9090/tcp
grafana       running   0.0.0.0:3000->3000/tcp
alertmanager  running   0.0.0.0:9093->9093/tcp
```

---

## API Reference

Base URL: `http://localhost:8000`

### `GET /health`

Returns the API status and the number of loaded models.

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "models_loaded": 8}
```

---

### `GET /models`

Lists all models registered in MLflow with their version and registration date.

```bash
curl http://localhost:8000/models
```

```json
[
  {"name": "yolov8", "version": "1", "registered_at": "2024-..."},
  {"name": "rtdetr",  "version": "1", "registered_at": "2024-..."}
]
```

---

### `POST /predict`

Runs inference on an uploaded image. Returns the waste detection result with a confidence score.

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@image.jpg" \
  -F "latitude=48.8566" \
  -F "longitude=2.3522" \
  -F "model_name=yolov8"
```

```json
{
  "rubbish": true,
  "confiance": 0.87,
  "model_used": "yolov8",
  "timestamp": "2024-..."
}
```

**Parameters**

| Field | Type | Description |
|---|---|---|
| `file` | `image/jpeg` or `image/png` | Image to analyse |
| `latitude` | float, [-90, 90] | GPS latitude |
| `longitude` | float, [-180, 180] | GPS longitude |
| `model_name` | string | One of the registered model names |

**Error responses**

| Code | Cause |
|---|---|
| `422` | Invalid file type (non-image) |
| `422` | GPS coordinates out of range |
| `422` | Unknown model name |

---

### `GET /history`

Returns all past detections stored in the database.

```bash
curl http://localhost:8000/history
```

---

### `GET /metrics`

Exposes Prometheus metrics for the inference service.

```bash
curl http://localhost:8000/metrics | grep "^ml_"
```

| Metric | Description |
|---|---|
| `ml_predictions_total` | Total number of predictions |
| `ml_inference_latency_seconds` | Inference latency histogram |
| `ml_predictions_by_model_total` | Predictions broken down by model |
| `ml_validation_errors_total` | Total input validation errors |

---

## Models

Eight object detection models are registered in MLflow and loaded at startup:

| Model | Description |
|---|---|
| `yolov8` | YOLOv8 baseline |
| `yolo26` | YOLOv26 variant |
| `rtdetr` | RT-DETR transformer detector |
| `rtdetrv2` | RT-DETRv2 |
| `rfdetr` | RF-DETR |
| `dfine` | D-FINE detector |
| `deim-dfine` | DEIM + D-FINE fusion |
| `fusion-model` | Ensemble fusion model |

The MLflow registry UI is accessible at `http://localhost:5000`.

---

## ETL Pipeline

Two Airflow DAGs handle automated drone data ingestion.

### DAG 1 — `drone_mission_simulator`

Simulates drone patrol missions on a 5-minute schedule. Generates synthetic detections and inserts them into `drone_patrol.db`.

```bash
# Check run history
docker compose exec airflow airflow dags list-runs \
  --dag-id drone_mission_simulator --output table

# Verify generated data
docker compose exec airflow \
  sqlite3 /data/drone_patrol.db \
  "SELECT COUNT(*) FROM drone_detections;"
```

### DAG 2 — `drone_patrol_sync`

Extracts, filters, and loads drone detections into the application database. Triggered automatically by DAG 1 via `TriggerDagRunOperator`.

| Task | Description |
|---|---|
| `extract` | Reads new detections from `drone_patrol.db` |
| `transform` | Filters detections with `confiance >= 0.65` |
| `load` | Inserts results into `app_detections.db` and sets `processed = 1` |

```bash
# Trigger manually if needed
docker compose exec airflow airflow dags trigger drone_patrol_sync

# Check task states for the latest run
RUN_ID=$(docker compose exec airflow airflow dags list-runs \
  --dag-id drone_patrol_sync --output json | python -m json.tool \
  | python -c "import sys,json; runs=json.load(sys.stdin); print(runs[0]['run_id'])")

docker compose exec airflow airflow tasks states-for-dag-run \
  drone_patrol_sync "$RUN_ID" --output table
```

The Airflow UI is accessible at `http://localhost:8080`.

---

## Dashboard

The Streamlit app is accessible at `http://localhost:8501`.

Features:
- Model selection dropdown populated dynamically from `GET /models`
- Image upload with GPS input — displays confidence score and model used
- Interactive Folium map of all historical detections
- Filters by source, model, and time period
- Color-coded markers: **red** for manual uploads, **orange** for drone patrol detections

---

## Observability

### Prometheus & Grafana

Prometheus scrapes the `/metrics` endpoint automatically. The Grafana dashboard is accessible at `http://localhost:3000` and includes 4 panels covering prediction volume, inference latency, per-model breakdown, and validation error rate.

The dashboard definition is versioned at `monitoring/grafana/dashboard.json`.

### Structured Logging

Every prediction is appended to `logs/predictions.jsonl` as a structured JSON entry:

```json
{
  "timestamp": "2024-...",
  "model_name": "yolov8",
  "confiance": 0.87,
  "source": "manual_upload",
  "latence_ms": 142
}
```

### Alerting

Alert rules are defined in `monitoring/alertmanager.yml` and loaded into Prometheus at startup.

```bash
# Check active rules
curl http://localhost:9090/api/v1/rules

# Check Alertmanager cluster status
curl http://localhost:9093/api/v2/status
```

---

## CI/CD

The GitHub Actions pipeline runs on every push to `main` and on every pull request:

1. Runs unit tests — `pytest api/tests/test_unit.py`
2. Runs integration tests — `pytest api/tests/test_integration.py`
3. Builds and pushes the Docker image to GitHub Container Registry

```bash
# Pull the latest published image
docker pull ghcr.io/gastlinassim/waste-api:latest
```

Pipeline history and logs: https://github.com/GASTLINASSIM/waste-detection-mlops-eceparis/actions

---

## Testing

```bash
# Unit tests
pytest api/tests/test_unit.py -v

# Integration tests (requires the stack to be running)
pytest api/tests/test_integration.py -v
```

---

## Project Structure

```
waste-detection-mlops-eceparis/
├── api/                          # FastAPI inference service
│   ├── Dockerfile
│   └── tests/
│       ├── test_unit.py
│       └── test_integration.py
├── app/                          # Streamlit dashboard
│   └── Dockerfile
├── dags/                         # Airflow DAGs
│   ├── drone_mission_simulator.py
│   └── drone_patrol_sync.py
├── monitoring/
│   ├── grafana/
│   │   └── dashboard.json
│   └── alertmanager.yml
├── logs/
│   └── predictions.jsonl
├── docker-compose.yml
├── generate_patrol_db.py
└── requirements.txt
```