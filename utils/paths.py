from pathlib import Path
import os

PROJECT_PATH = Path(__file__).resolve().parent.parent

DATA_PATH = os.path.join(PROJECT_PATH, "data", "ModelNet40")
MODEL_PATH = os.path.join(PROJECT_PATH, "models")

BEST_MODEL_PATH = os.path.join(MODEL_PATH, "pointnet_model_best.pth")
FINAL_MODEL_PATH = os.path.join(MODEL_PATH, "pointnet_model_final.pth")

BEST_ONNX_PATH = os.path.join(MODEL_PATH, "pointnet_model_best.onnx")

PATHS = {
    "Project": PROJECT_PATH,
    "Data": DATA_PATH,
    "Model": MODEL_PATH,
}

for name, path in PATHS.items():
    print(f"{name} Path: {path}")