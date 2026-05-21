# 🏥 MedHelp — Multiple Disease Recognition System

An advanced web-based clinical decision-support tool that predicts the risk of four
major diseases using **three ML models built entirely from scratch** (pure NumPy — no
scikit-learn, TensorFlow, or PyTorch).

---

## 📁 Project Structure

```
MedHelp/
├── app/
│   └── streamlit_app.py       # Streamlit web application
├── data/
│   ├── README.md              # Dataset download instructions
│   ├── diabetes.csv           # ← place Kaggle CSVs here
│   ├── heart.csv
│   ├── indian_liver_patient.csv
│   └── kidney_disease.csv
├── models/
│   └── *.sav                  # Trained model files (auto-generated)
├── utils/
│   ├── __init__.py
│   ├── logistic_regression.py # LR from scratch
│   ├── svm_classifier.py      # SVM from scratch
│   ├── neural_network.py      # NN from scratch
│   └── visualization.py       # Plotly charts
├── data_preprocessing.py      # Data loading & cleaning pipeline
├── train_models.py            # Model training & serialisation
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download datasets
Place the four Kaggle CSVs into the `/data` folder (see `data/README.md`).

### 3. Train the models
```bash
python train_models.py
```
This trains all 12 models (3 models × 4 diseases) and saves `.sav` files to `/models`.

### 4. Launch the app
```bash
streamlit run app/streamlit_app.py
```

---

## 🧠 ML Models (Built From Scratch)

### Logistic Regression
- Sigmoid activation: `σ(z) = 1 / (1 + e^{-z})`
- Binary Cross-Entropy loss
- Gradient Descent weight updates

### SVM Classifier
- Linear hyperplane: `w·x + b`
- Hinge Loss: `max(0, 1 - y(w·x + b))`
- Sub-gradient Descent with L2 regularisation

### Single-Layer Neural Network
- Architecture: Input → Dense(16, sigmoid) → Dense(1, sigmoid)
- Xavier weight initialisation
- Backpropagation + Mini-Batch Gradient Descent

---

## 🌐 Features

| Feature | Description |
|---|---|
| Sidebar Navigation | Select disease via `streamlit_option_menu` |
| Input Form | Columnar layout with number inputs & select boxes |
| Model Dashboard | Side-by-side comparison of all 3 models |
| Probability Scores | Raw sigmoid output displayed as % risk |
| Radar Chart | Patient metrics vs healthy baseline |
| Loss Curves | Training convergence visualisation |
| Bulk CSV Upload | Process multiple patients, download results |

---

## ☁️ Cloud Deployment

The app is ready for deployment on **Streamlit Community Cloud**:

1. Push the repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo.
3. Set the main file path to `app/streamlit_app.py`.
4. The `requirements.txt` handles all dependencies automatically.

> **Note:** Pre-train models locally and commit the `/models/*.sav` files to the repo,
> or add a training step to your deployment pipeline.

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**.
It is not a substitute for professional medical diagnosis or advice.
