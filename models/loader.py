# File: models/loader.py

import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(csv_path: str):
    """
    Load the dataset and split it into train and test sets.
    Returns: X_train, X_test, y_train, y_test
    """
    df = pd.read_csv(csv_path)
    df.dropna(subset=['document', 'summary'], inplace=True)
    return train_test_split(df['document'], df['summary'], test_size=0.2, random_state=42)
