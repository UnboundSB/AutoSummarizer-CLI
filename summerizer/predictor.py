# File: summerizer/predictor.py

import joblib
from models.summerizer_benchmark import TfidfSummaryModel
import os

MODEL_PATH = "models/summarizer_model.joblib"

def load_model():
    """
    Load the trained summarization model.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Trained model not found. Please run training first.")
    return joblib.load(MODEL_PATH)

def summarize(text: str) -> str:
    model = load_model()
    result = model.predict([text])[0]
    print(f"DEBUG: {result}")
    return result

