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


def predict_single(bundle: ModelBundle, features: Any) -> dict:
    """Run prediction for a single sample.

    `features` can be either a 1-D list of floats or a numpy array. If a numpy
    array with shape (1, n_features) is provided it will be used as-is. If a
    scaler is present it will be applied. The function will try `predict` and
    fall back to `predict_proba` (when using probabilities it will take argmax
    to obtain a class index before inverse-transforming).

    Returns dict with keys: prediction (decoded if label encoder present), raw (model output), success
    """
    if bundle.model is None:
        raise RuntimeError("No model loaded")

    # prepare numpy array input
    if isinstance(features, np.ndarray):
        arr = features
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
    else:
        arr = np.array(features, dtype=float).reshape(1, -1)

    if bundle.scaler is not None:
        arr = bundle.scaler.transform(arr)

    # predict may return encoded labels or probabilities depending on model
    used_proba = False
    try:
        raw_pred = bundle.model.predict(arr)
    except Exception:
        # try predict_proba if predict fails
        try:
            raw_pred = bundle.model.predict_proba(arr)
            used_proba = True
        except Exception:
            raise

    # make raw JSON-serializable when possible
    raw_for_out = raw_pred.tolist() if hasattr(raw_pred, "tolist") else raw_pred
    out = {"raw": raw_for_out}

    # decode if label encoder is available
    if bundle.label_encoder is not None:
        try:
            if used_proba:
                probs = np.asarray(raw_pred)
                class_idx = np.argmax(probs, axis=1)
                decoded = bundle.label_encoder.inverse_transform(class_idx)
                out["prediction"] = decoded.tolist()[0]
            else:
                arr_labels = np.asarray(raw_pred).ravel()
                decoded = bundle.label_encoder.inverse_transform(arr_labels)
                out["prediction"] = decoded.tolist()[0]
        except Exception:
            # fallback: return raw
            out["prediction"] = out["raw"]
    else:
        # no label encoder: if we used proba, return argmax index; else return raw value
        if used_proba:
            probs = np.asarray(raw_pred)
            class_idx = np.argmax(probs, axis=1)
            out["prediction"] = int(class_idx[0])
        else:
            arrp = np.asarray(raw_pred)
            if arrp.size == 1:
                out["prediction"] = arrp.tolist()[0]
            else:
                out["prediction"] = arrp.tolist()

    out["success"] = True
    return out
