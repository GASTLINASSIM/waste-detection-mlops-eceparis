import io
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Permet d'importer api/main.py depuis tests/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import app, MODELS

client = TestClient(app)


def _make_jpeg(width=640, height=640) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color="white").save(buf, format="JPEG")
    return buf.getvalue()


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_models_list_has_entries():
    """Sanity : au moins un modèle chargé (selon l'env)."""
    r = client.get("/models")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    # Si le test tourne sans MLflow, MODELS est vide — on skip plutôt que de faire échouer
    if not data:
        pytest.skip("Aucun modèle chargé (MLflow non accessible)")
    for m in data:
        assert "name" in m and "version" in m and "registered_at" in m


def test_predict_valid_image():
    if not MODELS:
        pytest.skip("Aucun modèle chargé")
    first_model = next(iter(MODELS))
    files = {"file": ("t.jpg", _make_jpeg(), "image/jpeg")}
    data = {"latitude": 48.85, "longitude": 2.35, "model_name": first_model}
    r = client.post("/predict", files=files, data=data)
    assert r.status_code == 200
    body = r.json()
    assert body["model_used"] == first_model
    assert 0.0 <= body["confiance"] <= 1.0


def test_predict_invalid_file():
    files = {"file": ("t.txt", b"pas une image", "text/plain")}
    data = {"latitude": 48.85, "longitude": 2.35, "model_name": "yolov8"}
    r = client.post("/predict", files=files, data=data)
    assert r.status_code == 422


def test_predict_invalid_gps():
    files = {"file": ("t.jpg", _make_jpeg(), "image/jpeg")}
    data = {"latitude": 999, "longitude": 2.35, "model_name": "yolov8"}
    r = client.post("/predict", files=files, data=data)
    assert r.status_code == 422


def test_predict_unknown_model():
    files = {"file": ("t.jpg", _make_jpeg(), "image/jpeg")}
    data = {"latitude": 48.85, "longitude": 2.35, "model_name": "modele_inexistant"}
    r = client.post("/predict", files=files, data=data)
    assert r.status_code == 422