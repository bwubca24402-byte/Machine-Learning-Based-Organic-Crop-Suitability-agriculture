"""Preprocessing helpers for Organic Crop Suitability Assessment."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = ["pH", "EC", "Organic_C", "Nitrogen", "Phosphorus", "Potassium", "Sulphur", "Zinc", "Iron", "Manganese", "Copper", "Boron"]
TARGET = "Organic_Farming_Suitability"

def load_dataset(path="organic_farming_suitability_1000_source_calibrated.csv"):
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError("Dataset is missing columns: " + ", ".join(missing))
    df = df[FEATURES + [TARGET]].copy()
    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=FEATURES + [TARGET])
    return df

def split_data(df, test_size=0.2, random_state=42):
    X, y = df[FEATURES], df[TARGET].astype(str)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def fit_scaler(X_train):
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler
