# File: summerizer/train.py

import joblib
import os
from models.loader import load_data
from models.summerizer_benchmark import TfidfSummaryModel

MODEL_PATH = "models/summarizer_model.joblib"

def train_and_save_model():
    """
    Train the summarization model and save it.
    """
    X_train, X_test, y_train, y_test = load_data("data/summaries.csv")
    model = TfidfSummaryModel()
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model trained and saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_and_save_model()
