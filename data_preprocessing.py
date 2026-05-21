"""
data_preprocessing.py
---------------------
Loads, cleans, and scales the four Kaggle disease datasets.

Expected CSV files in /data:
  - diabetes.csv          (Pima Indians Diabetes)
  - heart.csv             (UCI Heart Disease)
  - indian_liver_patient.csv
  - kidney_disease.csv    (Chronic Kidney Disease)

Outputs:
  - Cleaned DataFrames ready for model training
  - A fitted StandardScaler (implemented from scratch) per dataset
"""

import os
import numpy as np
import pandas as pd
import pickle

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Scratch Standard Scaler
# ---------------------------------------------------------------------------
class StandardScalerScratch:
    """
    Z-score normalisation: x_scaled = (x - μ) / σ
    Implemented without scikit-learn.
    """

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScalerScratch":
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Avoid division by zero for constant features
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        return X_scaled * self.std_ + self.mean_


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def _fill_missing_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Replace 0s that represent missing values and fill NaNs with column median."""
    # Columns where 0 is physiologically impossible → treat as NaN
    zero_invalid_cols = [
        "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
        "TotalBilirubin", "DirectBilirubin", "AlkalinePhosphotase",
        "AlamineAminotransferase", "AspartateAminotransferase",
        "TotalProteins", "Albumin", "AlbuminAndGlobulinRatio",
    ]
    for col in zero_invalid_cols:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)

    # Fill all remaining NaNs with column median
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

    return df


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode all object/category columns."""
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = df[col].astype("category").cat.codes
    return df


def train_test_split_scratch(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42
):
    """Simple train/test split without scikit-learn."""
    rng = np.random.default_rng(seed)
    n = len(y)
    indices = rng.permutation(n)
    split = int(n * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ---------------------------------------------------------------------------
# Dataset-specific loaders
# ---------------------------------------------------------------------------
def load_diabetes(path: str | None = None):
    """
    Pima Indians Diabetes Dataset.
    Target column: 'Outcome' (0 = no diabetes, 1 = diabetes)
    """
    path = path or os.path.join(DATA_DIR, "diabetes.csv")
    df = pd.read_csv(path)

    feature_cols = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
    ]
    target_col = "Outcome"

    df = _fill_missing_numeric(df)
    df = _encode_categoricals(df)

    X = df[feature_cols].values.astype(float)
    y = df[target_col].values.astype(float)

    scaler = StandardScalerScratch()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, feature_cols, scaler


def load_heart(path: str | None = None):
    """
    UCI Heart Disease Dataset (Cleveland / combined versions).

    Supported Kaggle sources:
      - https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci  (303 rows)
      - https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset     (1025 rows)

    Expected columns (standard 14-col Cleveland format):
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal, target

    Target column: 'target' (0 = no disease, 1 = disease).
    Multi-class targets (0-4) are binarised to 0/1.
    """
    path = path or os.path.join(DATA_DIR, "heart.csv")
    df = pd.read_csv(path)

    # Normalise column names: strip whitespace, lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    # Handle alternate target column names
    if "target" not in df.columns:
        if "condition" in df.columns:
            df = df.rename(columns={"condition": "target"})
        elif "num" in df.columns:
            # UCI raw format uses 'num' (0-4)
            df = df.rename(columns={"num": "target"})
        else:
            raise ValueError(
                "Could not find a target column in heart.csv. "
                "Expected 'target', 'condition', or 'num'."
            )

    # Binarise: any value > 0 means disease present
    df["target"] = (df["target"] > 0).astype(int)

    # Replace '?' placeholders (common in raw UCI files) with NaN
    df = df.replace("?", np.nan)

    df = _fill_missing_numeric(df)
    df = _encode_categoricals(df)

    target_col = "target"
    # Standard feature order matching the Streamlit input form
    preferred_order = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
    ]
    feature_cols = [c for c in preferred_order if c in df.columns]
    # Append any extra columns not in the preferred list
    feature_cols += [c for c in df.columns if c not in feature_cols and c != target_col]

    X = df[feature_cols].values.astype(float)
    y = df[target_col].values.astype(float)

    scaler = StandardScalerScratch()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, feature_cols, scaler


def load_liver(path: str | None = None):
    """
    Indian Liver Patient Records.
    Target column: 'Dataset' (1 = liver patient, 2 = not)
    Converted to binary: 1 = patient, 0 = healthy.
    """
    path = path or os.path.join(DATA_DIR, "indian_liver_patient.csv")
    df = pd.read_csv(path)

    # Standardise column names
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    # Encode Gender
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0}).fillna(0)

    # Binarise target: 1 → 1 (patient), 2 → 0 (healthy)
    target_col = "Dataset"
    df[target_col] = (df[target_col] == 1).astype(int)

    df = _fill_missing_numeric(df)
    df = _encode_categoricals(df)

    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].values.astype(float)
    y = df[target_col].values.astype(float)

    scaler = StandardScalerScratch()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, feature_cols, scaler


def load_kidney(path: str | None = None):
    """
    Chronic Kidney Disease Dataset.
    Target column: 'classification' (ckd / notckd → 1 / 0)
    """
    path = path or os.path.join(DATA_DIR, "kidney_disease.csv")
    df = pd.read_csv(path)

    # Drop id column if present
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Standardise column names
    df.columns = [c.strip().replace("\t", "").replace(" ", "_") for c in df.columns]

    # Binarise target
    target_col = "classification"
    df[target_col] = df[target_col].str.strip().map(
        {"ckd": 1, "notckd": 0, "ckd\t": 1}
    ).fillna(0).astype(int)

    # Replace '\t' artifacts in values
    df = df.replace(to_replace=r"\t", value="", regex=True)

    df = _fill_missing_numeric(df)
    df = _encode_categoricals(df)

    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].values.astype(float)
    y = df[target_col].values.astype(float)

    scaler = StandardScalerScratch()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, feature_cols, scaler


# ---------------------------------------------------------------------------
# Convenience mapping
# ---------------------------------------------------------------------------
DATASET_LOADERS = {
    "Diabetes": load_diabetes,
    "Heart Disease": load_heart,
    "Liver Disease": load_liver,
    "Kidney Disease": load_kidney,
}


if __name__ == "__main__":
    for name, loader in DATASET_LOADERS.items():
        try:
            X, y, cols, scaler = loader()
            print(f"[{name}] X={X.shape}, y={y.shape}, features={cols}")
        except FileNotFoundError as e:
            print(f"[{name}] CSV not found — {e}")
