import os
import pickle
from typing import Any, List

import numpy as np


class ModelBundle:
    """Holds loaded model, scaler, and label encoder."""

    def __init__(self, model: Any, scaler: Any = None, label_encoder: Any = None):
        self.model = model
        self.scaler = scaler
        self.label_encoder = label_encoder


def load_bundle(models_dir: str = None) -> ModelBundle:
    """Load model, scaler, and label encoder from a directory.

    Expected filenames (fall back if missing):
      - lgbm_model.pkl or model.pkl
      - scaler.pkl
      - le.pkl or label_encoder.pkl

    Returns a ModelBundle with any missing artifacts set to None.
    """
    if models_dir is None:
        base = os.path.dirname(__file__)
        models_dir = os.path.join(base, "models")

    model = None
    scaler = None
    le = None

    # try common model filenames
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


def predict_single(bundle: ModelBundle, features: List[float]) -> dict:
    """Run prediction for a single sample (1D list of features).

    Returns dict with keys: prediction (decoded if label encoder present), raw (model output), success
    """
    if bundle.model is None:
        raise RuntimeError("No model loaded")

    arr = np.array(features, dtype=float).reshape(1, -1)
    if bundle.scaler is not None:
        arr = bundle.scaler.transform(arr)

    # predict may return encoded labels or probabilities depending on model
    try:
        raw_pred = bundle.model.predict(arr)
    except Exception as e:
        # try predict_proba if predict fails
        try:
            raw_pred = bundle.model.predict_proba(arr)
        except Exception:
            raise

    out = {"raw": raw_pred.tolist() if hasattr(raw_pred, "tolist") else raw_pred}

    if bundle.label_encoder is not None:
        try:
            # handle both array-like and single-value
            if hasattr(raw_pred, "ravel"):
                decoded = bundle.label_encoder.inverse_transform(np.asarray(raw_pred).ravel())
                out["prediction"] = decoded.tolist()[0]
            else:
                out["prediction"] = bundle.label_encoder.inverse_transform([raw_pred])[0]
        except Exception:
            # fallback: return raw
            out["prediction"] = out["raw"]
    else:
        # no label encoder, return raw predictions
        out["prediction"] = out["raw"]

    out["success"] = True
    return out
