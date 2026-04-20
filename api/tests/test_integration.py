"""
Test d'intégration : envoie une vraie requête à l'API déjà démarrée.
Prérequis : l'API doit tourner sur localhost:8000 (uvicorn ou docker compose).
"""
import time
from pathlib import Path

import pytest
import requests

API_URL = "http://localhost:8000"


@pytest.fixture(scope="module")
def api_ready():
    """Vérifie que l'API est accessible."""
    for _ in range(30):
        try:
            r = requests.get(f"{API_URL}/health", timeout=2)
            if r.ok:
                return API_URL
        except Exception:
            time.sleep(2)
    pytest.fail("API non accessible sur localhost:8000")


def test_api_end_to_end(api_ready):
    img_path = Path(__file__).resolve().parents[2] / "test_image.jpg"
    if not img_path.exists():
        pytest.skip(f"test_image.jpg introuvable à {img_path}")

    with open(img_path, "rb") as f:
        files = {"file": ("test_image.jpg", f.read(), "image/jpeg")}
    data = {"latitude": 48.8566, "longitude": 2.3522, "model_name": "yolov8"}

    r = requests.post(f"{api_ready}/predict", files=files, data=data, timeout=60)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_used"] == "yolov8"
    assert "confiance" in body


def test_history_returns_list(api_ready):
    r = requests.get(f"{api_ready}/history", timeout=10)
    assert r.status_code == 200
    assert isinstance(r.json(), list)