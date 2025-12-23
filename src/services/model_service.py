import pathlib
import pandas as pd
import logging
from fastapi import HTTPException, UploadFile, File
from src.models.model_bundle import load_bundle

class ModelService:
    def __init__(self):
        self._model = None
        self._model_path = None

    def _get_model(self):
        if self._model is not None:
            return self._model
        # Find model file
        repo_file = pathlib.Path(__file__).resolve()
        found = None
        for p in repo_file.parents:
            candidate = p / ".." / "models" / "lgbm_model.pkl"
            if candidate.exists():
                found = candidate
                break
        if not found:
            raise HTTPException(status_code=500, detail="Model file not found")
        self._model_path = found
        self._model = load_bundle(str(found.parent))
        return self._model

    async def predict(self, file: UploadFile = File(...)):
        raw = await file.read()
        try:
            text = raw.decode("utf-8")
        except Exception:
            text = raw.decode("latin-1")
        from io import StringIO
        try:
            df = pd.read_csv(StringIO(text))
        except Exception:
            raise HTTPException(status_code=400, detail="Failed to parse CSV file")
        if df.shape[0] == 0:
            raise HTTPException(status_code=400, detail="Empty CSV file")
        if "id" not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain an 'id' column")
        ids = df["id"].tolist()
        X_df = df.drop(columns=["id"])
        try:
            X = X_df.astype(float)
        except Exception:
            try:
                X = X_df.apply(pd.to_numeric, errors="raise")
            except Exception:
                raise HTTPException(status_code=400, detail="CSV contains non-numeric values in feature columns")
        model_bundle = self._get_model()
        label_map = {0: "CANDIDATE", 1: "CONFIRMED", 2: "FALSE POSITIVE"}
        results = []
        for idx, row in X.iterrows():
            feats = row.tolist()
            try:
                preds = model_bundle.model.predict([feats])
                pred0 = preds[0]
            except Exception as e:
                logging.error("Model prediction failed for row %s: %s", idx, e)
                try:
                    transit_depth = df.at[idx, "transit_depth"]
                except Exception:
                    transit_depth = None
                try:
                    orbital_period = df.at[idx, "orbital_period"]
                except Exception:
                    orbital_period = None
                results.append({"id": ids[idx], "transit_depth": transit_depth, "orbital_period": orbital_period, "transit_duration": None, "label": None})
                continue
            try:
                lab = label_map.get(int(pred0), str(pred0))
            except Exception:
                lab = str(pred0)
            try:
                transit_depth = df.at[idx, "transit_depth"]
            except Exception:
                transit_depth = None
            try:
                orbital_period = df.at[idx, "orbital_period"]
            except Exception:
                orbital_period = None
            results.append({"id": ids[idx], "transit_depth": transit_depth, "orbital_period": orbital_period, "label": lab})
        return {"predictions": results}
