import os
import pickle
from typing import Any

class ModelBundle:
    def __init__(self, model: Any, scaler: Any = None, label_encoder: Any = None):
        self.model = model
        self.scaler = scaler
        self.label_encoder = label_encoder

def load_bundle(models_dir: str = None) -> ModelBundle:
    if models_dir is None:
        base = os.path.dirname(__file__)
        models_dir = os.path.join(base, "models")
    model = None
    scaler = None
    le = None
    model_paths = [os.path.join(models_dir, n) for n in ("lgbm_model.pkl", "model.pkl")]
    for p in model_paths:
        if os.path.exists(p):
            with open(p, "rb") as f:
                model = pickle.load(f)
            break
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    le_paths = [os.path.join(models_dir, n) for n in ("le.pkl", "label_encoder.pkl")]
    for p in le_paths:
        if os.path.exists(p):
            with open(p, "rb") as f:
                le = pickle.load(f)
            break
    return ModelBundle(model=model, scaler=scaler, label_encoder=le)
