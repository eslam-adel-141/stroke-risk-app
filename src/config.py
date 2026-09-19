import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # reads .env when running locally; harmless if the file doesn't exist

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(BASE_DIR / "models")))
APP_TITLE = os.getenv("APP_TITLE", "Stroke Risk Predictor")
RISK_THRESHOLD = float(os.getenv("RISK_THRESHOLD", "50"))  # At Risk = 1 when risk >= 50%