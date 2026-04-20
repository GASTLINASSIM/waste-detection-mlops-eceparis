"""
Enregistre les 8 modèles dans le MLflow Registry.
Lancer avec le serveur MLflow http://localhost:5000.
"""
import torch 
from pathlib import Path
import numpy as np
from PIL import Image
import mlflow
import mlflow.pyfunc

mlflow.set_tracking_uri("http://localhost:5000")
WEIGHTS_DIR = Path("models")


class UltralyticsWrapper(mlflow.pyfunc.PythonModel):
    """Pour yolov8, yolo26, rtdetr, rfdetr, fusion-model."""

    def __init__(self, cls: str = "YOLO"):
        self.cls = cls  # "YOLO" ou "RTDETR"

    def load_context(self, context):
        from ultralytics import YOLO, RTDETR
        weights_path = context.artifacts["weights"]
        self.model = YOLO(weights_path) if self.cls == "YOLO" else RTDETR(weights_path)

    def predict(self, context, model_input, params=None):
        arr = model_input[0] if isinstance(model_input, list) else model_input
        if isinstance(arr, np.ndarray) is False:
            arr = np.array(arr)
        results = self.model.predict(arr, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return {"rubbish": False, "confiance": 0.0}
        conf = float(boxes.conf.max().item())
        return {"rubbish": conf > 0.5, "confiance": conf}


class HFDetectionWrapper(mlflow.pyfunc.PythonModel):
    """Pour rtdetrv2, dfine, deim-dfine (HuggingFace)."""

    def __init__(self, hf_id: str):
        self.hf_id = hf_id

    def load_context(self, context):
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        self.processor = AutoImageProcessor.from_pretrained(self.hf_id)
        self.model = AutoModelForObjectDetection.from_pretrained(self.hf_id)

    def predict(self, context, model_input, params=None):
        import torch
        arr = model_input[0] if isinstance(model_input, list) else model_input
        inputs = self.processor(images=arr, return_tensors="pt")
        with torch.no_grad():
            out = self.model(**inputs)
        scores = out.logits.softmax(-1)[..., :-1].max(-1).values
        conf = float(scores.max().item()) if scores.numel() else 0.0
        return {"rubbish": conf > 0.5, "confiance": conf}


def resolve_weights(filename: str) -> str:
    """Retourne le chemin des poids (télécharge yolov8n.pt si besoin)."""
    if filename is None:
        return None

    p = WEIGHTS_DIR / filename
    if p.exists():
        return str(p.resolve())

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    fallback_path = WEIGHTS_DIR / "yolov8n_fallback.pt"

    if not fallback_path.exists():
        print(f"   ⚠ {filename} absent, téléchargement de yolov8n.pt...")
        from ultralytics import YOLO
        # YOLO('yolov8n.pt') télécharge auto les poids pré-entraînés
        _ = YOLO("yolov8n.pt")
        # Le fichier est téléchargé dans le cwd, on le déplace
        import shutil
        if Path("yolov8n.pt").exists():
            shutil.move("yolov8n.pt", fallback_path)
            print(f"   ✓ Téléchargé vers {fallback_path}")
        else:
            raise FileNotFoundError("Échec du téléchargement de yolov8n.pt")
    else:
        print(f"   ⚠ {filename} absent, fallback {fallback_path}")

    return str(fallback_path.resolve())


MODELS = [
    # (nom_registry, wrapper, kwargs, fichier_poids)
    ("waste-detector-yolov8",       UltralyticsWrapper, {"cls": "YOLO"},   "yolov8n.pt"),
    ("waste-detector-yolo26",       UltralyticsWrapper, {"cls": "YOLO"},   "yolo26n.pt"),
    ("waste-detector-rtdetr",       UltralyticsWrapper, {"cls": "RTDETR"}, "rtdetr-s.pt"),
    ("waste-detector-rtdetrv2",     HFDetectionWrapper, {"hf_id": "PekingU/rtdetr_v2_r18vd"}, None),
    ("waste-detector-rfdetr",       UltralyticsWrapper, {"cls": "YOLO"},   "rfdetr.pt"),
    ("waste-detector-dfine",        HFDetectionWrapper, {"hf_id": "ustc-community/dfine-nano-coco"}, None),
    ("waste-detector-deim-dfine",   HFDetectionWrapper, {"hf_id": "ustc-community/dfine-nano-coco"}, None),
    ("waste-detector-fusion-model", UltralyticsWrapper, {"cls": "RTDETR"}, "yolov8n_yolo_neck_rtdetr_head.yaml"),
]


def main():
    client = mlflow.MlflowClient()
    sample_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    for registry_name, wrapper_cls, kwargs, weight_file in MODELS:
        print(f"\n→ Enregistrement {registry_name}")
        artifacts = {}
        if weight_file:
            artifacts["weights"] = resolve_weights(weight_file)

        with mlflow.start_run(run_name=f"register_{registry_name}"):
            if wrapper_cls is UltralyticsWrapper:
                model = wrapper_cls(**kwargs)
            else:
                model = wrapper_cls(**kwargs)

            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=model,
                artifacts=artifacts if artifacts else None,
                registered_model_name=registry_name,
                pip_requirements=[
                    "mlflow", "torch", "ultralytics",
                    "transformers", "pillow", "numpy",
                ],
            )

        versions = client.search_model_versions(f"name='{registry_name}'")
        latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
        client.transition_model_version_stage(
            name=registry_name,
            version=latest.version,
            stage="Production",
            archive_existing_versions=True,
        )
        print(f"   ✓ v{latest.version} → Production")

    print("\n✅ 8 modèles enregistrés")


if __name__ == "__main__":
    main()