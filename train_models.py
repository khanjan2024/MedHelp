"""
train_models.py
---------------
Trains all three scratch-built models on each disease dataset
and serialises them as .sav files using pickle.

Run from the project root:
    python train_models.py

Output files in /models:
    {disease}_{model}.sav       — trained model object
    {disease}_scaler.sav        — fitted StandardScalerScratch
    {disease}_features.sav      — list of feature column names
    {disease}_metrics.sav       — dict of accuracy scores
"""

import os
import pickle
import numpy as np

from data_preprocessing import DATASET_LOADERS, train_test_split_scratch
from utils.logistic_regression import LogisticRegressionScratch
from utils.svm_classifier import SVMClassifierScratch
from utils.neural_network import NeuralNetworkScratch

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Hyperparameters (tuned for reasonable convergence on these datasets)
# ---------------------------------------------------------------------------
LR_PARAMS = {"learning_rate": 0.1, "n_iterations": 500}
SVM_PARAMS = {"learning_rate": 0.001, "lambda_param": 0.01, "n_iterations": 300}
NN_PARAMS = {"hidden_size": 16, "learning_rate": 0.05, "n_epochs": 150, "batch_size": 32}


def save(obj, filename: str) -> None:
    path = os.path.join(MODELS_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  Saved → {path}")


def train_for_disease(disease_name: str, loader_fn) -> None:
    print(f"\n{'='*60}")
    print(f"  Training models for: {disease_name}")
    print(f"{'='*60}")

    # Load & preprocess
    try:
        X, y, feature_cols, scaler = loader_fn()
    except FileNotFoundError as e:
        print(f"  [SKIP] Dataset not found: {e}")
        return

    print(f"  Dataset shape: X={X.shape}, y={y.shape}")
    print(f"  Class balance: {int(y.sum())} positive / {len(y)} total")

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split_scratch(X, y, test_size=0.2)

    metrics = {}

    # ------------------------------------------------------------------
    # 1. Logistic Regression
    # ------------------------------------------------------------------
    print("\n  [1/3] Logistic Regression...")
    lr_model = LogisticRegressionScratch(**LR_PARAMS)
    lr_model.fit(X_train, y_train)
    acc = lr_model.accuracy(X_test, y_test)
    metrics["LogisticRegression"] = round(acc * 100, 2)
    print(f"        Test Accuracy: {acc*100:.2f}%")
    save(lr_model, f"{disease_name.replace(' ', '_')}_logistic_regression.sav")

    # ------------------------------------------------------------------
    # 2. SVM
    # ------------------------------------------------------------------
    print("\n  [2/3] SVM Classifier...")
    svm_model = SVMClassifierScratch(**SVM_PARAMS)
    svm_model.fit(X_train, y_train)
    acc = svm_model.accuracy(X_test, y_test)
    metrics["SVM"] = round(acc * 100, 2)
    print(f"        Test Accuracy: {acc*100:.2f}%")
    save(svm_model, f"{disease_name.replace(' ', '_')}_svm.sav")

    # ------------------------------------------------------------------
    # 3. Neural Network
    # ------------------------------------------------------------------
    print("\n  [3/3] Neural Network...")
    nn_model = NeuralNetworkScratch(**NN_PARAMS)
    nn_model.fit(X_train, y_train)
    acc = nn_model.accuracy(X_test, y_test)
    metrics["NeuralNetwork"] = round(acc * 100, 2)
    print(f"        Test Accuracy: {acc*100:.2f}%")
    save(nn_model, f"{disease_name.replace(' ', '_')}_neural_network.sav")

    # ------------------------------------------------------------------
    # Save scaler, feature list, and metrics
    # ------------------------------------------------------------------
    save(scaler, f"{disease_name.replace(' ', '_')}_scaler.sav")
    save(feature_cols, f"{disease_name.replace(' ', '_')}_features.sav")
    save(metrics, f"{disease_name.replace(' ', '_')}_metrics.sav")

    print(f"\n  Summary for {disease_name}:")
    for model_name, acc_val in metrics.items():
        print(f"    {model_name:25s}: {acc_val}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("MediPredictML — Model Training Pipeline")
    print("Ensure your CSV files are placed in the /data folder.\n")

    for disease, loader in DATASET_LOADERS.items():
        train_for_disease(disease, loader)

    print("\n\nAll models trained and saved to /models.")
    print("You can now launch the Streamlit app:  streamlit run app/streamlit_app.py")
