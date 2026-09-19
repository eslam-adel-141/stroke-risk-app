import json

import joblib
import pandas as pd

from .config import MODEL_DIR


class StrokeRiskPredictor:
    """Loads the saved regression + classification models and runs predictions."""

    def __init__(self, model_dir=MODEL_DIR):
        self.reg_model = joblib.load(model_dir / "reg_model.joblib")
        self.clf_model = joblib.load(model_dir / "clf_model.joblib")
        with open(model_dir / "metadata.json") as f:
            self.meta = json.load(f)
        self.features = self.meta["features"]
        self.symptoms = [f for f in self.features if f != "Age"]

    def predict(self, inputs: dict) -> dict:
        # Build a one-row DataFrame in the exact column order used in training
        X = pd.DataFrame([[inputs[f] for f in self.features]], columns=self.features)
        risk = max(0.0, min(100.0, float(self.reg_model.predict(X)[0])))
        label = int(self.clf_model.predict(X)[0])
        proba = float(self.clf_model.predict_proba(X)[0][1])
        return {"risk_percent": risk, "at_risk": label, "probability": proba}