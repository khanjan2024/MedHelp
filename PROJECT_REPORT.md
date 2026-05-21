# MedHelp — Multiple Disease Recognition System
## Project Report

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Objectives](#2-objectives)
3. [System Architecture](#3-system-architecture)
4. [Datasets](#4-datasets)
5. [Data Preprocessing Pipeline](#5-data-preprocessing-pipeline)
6. [Machine Learning Models](#6-machine-learning-models)
7. [Web Application](#7-web-application)
8. [Unique Features](#8-unique-features)
9. [Technology Stack](#9-technology-stack)
10. [Project Structure](#10-project-structure)
11. [Results & Evaluation](#11-results--evaluation)
12. [Limitations & Future Work](#12-limitations--future-work)
13. [Conclusion](#13-conclusion)

---

## 1. Project Overview

**MedHelp** is an advanced, web-based Multiple Disease Recognition System built using
Python, NumPy, and Streamlit. The system predicts the risk of four major diseases —
Diabetes, Heart Disease, Liver Disease, and Chronic Kidney Disease — by running patient
data simultaneously through three machine learning models that are implemented entirely
from scratch, without using any high-level ML libraries such as scikit-learn, TensorFlow,
or PyTorch.

The application is designed to serve as a clinical decision-support tool for healthcare
professionals and researchers. It provides not just binary predictions, but also raw
probability scores, model agreement analysis, risk tier classification, and interactive
data visualisations.

---

## 2. Objectives

- Build three core ML classifiers (Logistic Regression, SVM, Neural Network) from
  mathematical first principles using only NumPy.
- Train these models on real, publicly available medical datasets from Kaggle.
- Develop a professional, interactive web interface using Streamlit that allows
  single-patient prediction and bulk CSV processing.
- Provide interpretable outputs including probability scores, risk tiers, and
  model consensus analysis.
- Structure the codebase for cloud deployment readiness.

---

## 3. System Architecture

```
User Input (Web Form / CSV Upload)
            │
            ▼
   Data Preprocessing Layer
   (StandardScalerScratch, align_and_scale)
            │
            ▼
  ┌─────────┬──────────┬──────────────┐
  │  Logistic│   SVM    │   Neural     │
  │Regression│Classifier│   Network    │
  └─────────┴──────────┴──────────────┘
            │
            ▼
   Consensus Analysis Engine
   (Risk Tier · Agreement · Spread)
            │
            ▼
   Streamlit Dashboard
   (Results · Charts · Download)
```

The system follows a clean separation of concerns:
- `/utils` — all ML model implementations
- `/data` — raw Kaggle CSV datasets
- `/models` — serialised trained model files (.sav)
- `/app` — Streamlit frontend
- Root scripts — preprocessing and training pipelines

---

## 4. Datasets

Four publicly available medical datasets from Kaggle are used:

| Disease | Dataset | Source | Rows | Features | Target |
|---|---|---|---|---|---|
| Diabetes | Pima Indians Diabetes | UCI / Kaggle | 768 | 8 | Outcome (0/1) |
| Heart Disease | UCI Heart Disease (Cleveland) | Kaggle | 303–1025 | 13 | target (0/1) |
| Liver Disease | Indian Liver Patient Records | UCI / Kaggle | 583 | 10 | Dataset (1/2) |
| Kidney Disease | Chronic Kidney Disease | UCI / Kaggle | 400 | 24 | classification |

### Dataset Descriptions

**Pima Indians Diabetes Dataset**
Contains diagnostic measurements for female patients of Pima Indian heritage. Features
include glucose concentration, blood pressure, BMI, insulin levels, and age. The target
variable indicates whether the patient tested positive for diabetes within five years.

**UCI Heart Disease Dataset**
The Cleveland version contains 13 clinical attributes including chest pain type, resting
blood pressure, serum cholesterol, maximum heart rate, and ST depression. The target
indicates presence or absence of heart disease.

**Indian Liver Patient Records**
Contains biochemical test results from patients in Andhra Pradesh, India. Features
include bilirubin levels, enzyme concentrations, and protein levels. The target
distinguishes liver patients from healthy individuals.

**Chronic Kidney Disease Dataset**
Contains 24 features including blood pressure, specific gravity, albumin, blood urea,
serum creatinine, haemoglobin, and various cell counts. The target classifies patients
as having CKD or not.

---

## 5. Data Preprocessing Pipeline

All preprocessing is implemented from scratch in `data_preprocessing.py`.

### 5.1 StandardScalerScratch

Z-score normalisation is applied to all feature matrices:

```
x_scaled = (x - μ) / σ
```

Where μ is the column mean and σ is the column standard deviation computed over the
training set. Constant features (σ = 0) are assigned σ = 1 to prevent division by zero.
The fitted scaler is serialised alongside each model so that inference-time inputs are
scaled identically to training data.

### 5.2 Missing Value Handling

Two strategies are applied:

1. **Zero-replacement**: Physiologically impossible zero values in columns such as
   Glucose, BloodPressure, BMI, and Insulin are replaced with NaN before imputation.
2. **Median imputation**: All remaining NaN values are filled with the column median,
   which is robust to outliers compared to mean imputation.

### 5.3 Categorical Encoding

All object and category dtype columns are label-encoded using Pandas category codes.
The Gender column in the Liver dataset is explicitly mapped (Male → 1, Female → 0).
The Kidney Disease target column handles tab-character artifacts (`ckd\t`) from the
raw UCI file.

### 5.4 Train/Test Split

An 80/20 stratified-style split is implemented from scratch using NumPy's
`default_rng(seed=42).permutation()` for reproducibility.

### 5.5 Feature Alignment

A key preprocessing step at inference time is the `align_and_scale()` function. Since
the web form collects only the most clinically relevant features (e.g., 18 out of 24
kidney disease features), this function:

1. Loads the saved feature column list from training.
2. Builds a full-length vector matching the scaler's expected shape.
3. Fills form-collected features by name-matching (case-insensitive).
4. Fills uncollected features with the training mean (which becomes 0 after scaling —
   the safest neutral imputation).

---

## 6. Machine Learning Models

All three models are implemented in the `/utils` directory using only Python and NumPy.
No external ML libraries are used for the core algorithms.

---

### 6.1 Logistic Regression (`logistic_regression.py`)

**Mathematical Foundation**

Logistic Regression models the probability of a binary outcome using the sigmoid
activation function applied to a linear combination of features:

```
ŷ = σ(w·x + b) = 1 / (1 + e^{-z})
```

**Loss Function — Binary Cross-Entropy (Log Loss)**

```
L = -1/N × Σ [ y·log(ŷ) + (1-y)·log(1-ŷ) ]
```

Numerical stability is ensured by clipping predictions to [1e-15, 1-1e-15] to
prevent log(0) errors.

**Gradient Descent Update Rules**

```
dL/dw = (1/N) × Xᵀ(ŷ - y)
dL/db = (1/N) × Σ(ŷ - y)

w ← w - α × dL/dw
b ← b - α × dL/db
```

**Hyperparameters**

| Parameter | Value |
|---|---|
| Learning Rate (α) | 0.1 |
| Iterations | 500 |
| Weight Initialisation | Zeros |

**Output**: Raw sigmoid probability used directly as the risk percentage displayed
in the UI.

---

### 6.2 SVM Classifier (`svm_classifier.py`)

**Mathematical Foundation**

The linear SVM finds a hyperplane w·x + b = 0 that maximises the margin between
classes. The soft-margin primal objective is:

```
min  (λ/2)||w||² + (1/N) Σ max(0, 1 - yᵢ(w·xᵢ + b))
```

Labels are internally converted from {0, 1} to {-1, +1} for the SVM formulation.

**Loss Function — Hinge Loss with L2 Regularisation**

```
L = (λ/2)||w||² + (1/N) Σ max(0, 1 - yᵢ(w·xᵢ + b))
```

**Sub-Gradient Descent Update Rules**

For each sample i:
- If yᵢ(w·xᵢ + b) ≥ 1 (correctly classified, outside margin):
  ```
  dw = λ·w,   db = 0
  ```
- Otherwise (inside margin or misclassified):
  ```
  dw = λ·w - yᵢ·xᵢ,   db = -yᵢ
  ```

**Hyperparameters**

| Parameter | Value |
|---|---|
| Learning Rate | 0.001 |
| Lambda (regularisation) | 0.01 |
| Iterations | 300 |

**Probability Output**: The raw decision function score w·x + b is passed through
a sigmoid to produce an approximate probability for display purposes.

---

### 6.3 Neural Network (`neural_network.py`)

**Architecture**

```
Input Layer (n features)
      ↓
Hidden Layer (16 neurons, Sigmoid activation)
      ↓
Output Layer (1 neuron, Sigmoid activation)
```

**Weight Initialisation — Xavier / Glorot Uniform**

```
limit = √(6 / (fan_in + fan_out))
W ~ Uniform(-limit, +limit)
```

This prevents vanishing/exploding gradients during early training.

**Forward Computation**

```
Z1 = X·W1 + b1
A1 = σ(Z1)          ← hidden layer activations
Z2 = A1·W2 + b2
A2 = σ(Z2)          ← output probability
```

**Loss Function — Binary Cross-Entropy**

```
L = -1/N × Σ [ y·log(A2) + (1-y)·log(1-A2) ]
```

**Backpropagation**

```
Output layer:
  dZ2 = A2 - y
  dW2 = (1/N) × A1ᵀ · dZ2
  db2 = mean(dZ2)

Hidden layer:
  dA1 = dZ2 · W2ᵀ
  dZ1 = dA1 ⊙ σ'(A1)     where σ'(a) = a(1-a)
  dW1 = (1/N) × Xᵀ · dZ1
  db1 = mean(dZ1)
```

**Mini-Batch Gradient Descent**

Data is shuffled at the start of each epoch and processed in batches of 32 samples.
This provides a balance between the noisy updates of stochastic GD and the slow
convergence of full-batch GD.

**Hyperparameters**

| Parameter | Value |
|---|---|
| Hidden Size | 16 neurons |
| Learning Rate | 0.05 |
| Epochs | 150 |
| Batch Size | 32 |

---

## 7. Web Application

The frontend is built with Streamlit and `streamlit_option_menu`. The app runs at
`http://localhost:8501` and is structured as a multi-page dashboard.

### 7.1 Sidebar Navigation

A persistent sidebar provides navigation between:
- Home (overview and status dashboard)
- Diabetes Prediction
- Heart Disease Prediction
- Liver Disease Prediction
- Kidney Disease Prediction

Training accuracy metrics for each disease are displayed in the sidebar once models
are trained.

### 7.2 Patient Input Forms

Each disease page presents a clean, three-column input form using `st.columns`.
Inputs include `st.number_input` for continuous variables and `st.selectbox` for
categorical variables (sex, chest pain type, ECG results, etc.).

### 7.3 Prediction Pipeline

On clicking "Run Prediction":
1. Form values are collected into a named dictionary.
2. `align_and_scale()` builds the full feature vector and applies the saved scaler.
3. All three models run simultaneously on the scaled input.
4. Results are displayed in the Model Comparison Dashboard.

### 7.4 Model Comparison Dashboard

Three side-by-side cards display each model's:
- Binary verdict (Positive / Negative)
- Raw probability percentage from the sigmoid output
- A visual progress bar gauge

A Plotly bar chart compares all three probability scores side by side.

### 7.5 Consensus Analysis Panel

A unique three-card panel below the model results provides:

**Risk Tier Badge** — classifies the overall result into one of four tiers:
- 🟢 Low Risk (avg probability < 30%, unanimous)
- 🟡 Moderate Risk (30–55%, unanimous)
- 🔴 High Risk (> 55%, unanimous)
- ⚠️ Uncertain (models disagree)

**Model Agreement Card** — shows the vote breakdown (e.g., "2 Positive · 1 Negative")
and the consensus percentage.

**Confidence Spread Card** — shows the maximum deviation between model probabilities
(±%), classified as Tight / Moderate / Wide.

**Uncertain Case Alert** — when models disagree, a highlighted warning box explains
the disagreement in plain language and recommends professional consultation.

### 7.6 Data Visualisation

**Radar Chart**: A Plotly polar chart overlays the patient's input values against
healthy population baselines for each disease. Both traces are normalised to [0, 1]
relative to the maximum value per feature.

**Training Loss Curves**: Three Plotly line charts show the convergence of each model's
loss function over training iterations/epochs.

### 7.7 Bulk CSV Upload

A `st.file_uploader` widget accepts CSV files containing multiple patient records.
The app:
1. Loads and displays the uploaded data.
2. Applies the saved scaler.
3. Runs all three models on every row.
4. Appends prediction and probability columns to the dataframe.
5. Provides a `st.download_button` to export the results as a new CSV.

---

## 8. Unique Features

### 8.1 Consensus Analysis Engine

The most distinctive feature of MedHelp is its ability to interpret disagreement
between models. Rather than presenting three independent results and leaving the
user to reconcile them, the system:

- Computes a **confidence spread score** (max deviation from mean probability)
- Assigns a **risk tier** that factors in both probability magnitude and model agreement
- Triggers an **Uncertain Case Alert** with a plain-language explanation when models
  conflict

This is particularly valuable in clinical contexts where a split verdict (e.g., 2
models positive, 1 negative) is more informative than a simple majority vote.

### 8.2 From-Scratch ML Implementation

All three models are implemented using only NumPy, with no dependency on scikit-learn,
TensorFlow, or PyTorch. This provides full transparency into the mathematical operations
and makes the codebase an educational resource as well as a functional tool.

### 8.3 Robust Feature Alignment

The `align_and_scale()` function solves a real engineering problem: the web form
collects a subset of features, but the trained model expects the full feature vector.
The alignment function handles this gracefully using name-based matching and
mean-imputation for unseen features.

---

## 9. Technology Stack

| Component | Technology | Version |
|---|---|---|
| Language | Python | 3.11 |
| Numerical Computing | NumPy | 1.26.4 |
| Data Manipulation | Pandas | 2.2.2 |
| Web Framework | Streamlit | 1.35.0 |
| Navigation | streamlit-option-menu | 0.3.13 |
| Visualisation | Plotly | 5.22.0 |
| Visualisation | Matplotlib | 3.9.0 |
| Model Serialisation | pickle | stdlib |
| ML Libraries | None (from scratch) | — |

---

## 10. Project Structure

```
MedHelp/
│
├── app/
│   └── streamlit_app.py          # Streamlit web application (frontend)
│
├── data/
│   ├── README.md                 # Dataset download instructions
│   ├── diabetes.csv              # Pima Indians Diabetes
│   ├── heart.csv                 # UCI Heart Disease
│   ├── indian_liver_patient.csv  # Indian Liver Patient Records
│   └── kidney_disease.csv        # Chronic Kidney Disease
│
├── models/
│   ├── Diabetes_logistic_regression.sav
│   ├── Diabetes_svm.sav
│   ├── Diabetes_neural_network.sav
│   ├── Diabetes_scaler.sav
│   ├── Diabetes_features.sav
│   ├── Diabetes_metrics.sav
│   └── ... (same pattern for Heart, Liver, Kidney)
│
├── utils/
│   ├── __init__.py
│   ├── logistic_regression.py    # LR from scratch
│   ├── svm_classifier.py         # SVM from scratch
│   ├── neural_network.py         # Neural Network from scratch
│   └── visualization.py          # Plotly chart functions
│
├── data_preprocessing.py         # Data loading, cleaning, scaling
├── train_models.py               # Model training and serialisation
├── requirements.txt              # Python dependencies
├── README.md                     # Setup and usage guide
└── PROJECT_REPORT.md             # This document
```

---

## 11. Results & Evaluation

Models are evaluated on a held-out 20% test split. The following accuracy ranges
are typical for these datasets with linear/shallow models:

| Disease | Logistic Regression | SVM | Neural Network |
|---|---|---|---|
| Diabetes | ~75–78% | ~72–76% | ~74–78% |
| Heart Disease | ~80–85% | ~78–83% | ~80–84% |
| Liver Disease | ~70–74% | ~68–72% | ~70–75% |
| Kidney Disease | ~92–96% | ~90–94% | ~92–96% |

> Note: Exact values depend on the specific Kaggle dataset version used and are
> printed to the console during `python train_models.py`. They are also displayed
> in the app sidebar after training.

Kidney Disease achieves the highest accuracy because the dataset has strong, clearly
separable features (creatinine, haemoglobin) and relatively low noise. Liver Disease
is the most challenging due to class imbalance (approximately 71% positive cases).

---

## 12. Limitations & Future Work

### Current Limitations

- **Linear SVM**: The scratch SVM uses a linear kernel. Non-linear kernels (RBF, polynomial)
  would improve performance on non-linearly separable data but require kernel trick
  implementation.
- **Shallow Neural Network**: A single hidden layer of 16 neurons is intentionally
  simple. Deeper architectures would improve accuracy but increase training time.
- **No cross-validation**: Models are evaluated on a single train/test split. K-fold
  cross-validation would give more reliable accuracy estimates.
- **Class imbalance**: No oversampling (SMOTE) or class weighting is applied, which
  may bias models toward the majority class on imbalanced datasets like Liver Disease.
- **SVM training speed**: The scratch SVM iterates over each sample per iteration,
  making it slow on large datasets. Vectorised updates would significantly speed this up.

### Future Enhancements

- Add SHAP-style feature importance explanations per prediction
- Implement K-fold cross-validation in the training pipeline
- Add a confidence calibration step (Platt scaling) for the SVM probability output
- Support additional diseases (Parkinson's, Breast Cancer)
- Add user authentication for clinical deployment
- Implement model retraining from the UI when new data is uploaded

---

## 13. Conclusion

MedHelp demonstrates that production-quality machine learning applications can be
built from mathematical first principles without relying on high-level ML frameworks.
The three models — Logistic Regression, SVM, and Neural Network — are implemented
transparently using NumPy, making every mathematical operation visible and auditable.

The Streamlit web application provides a professional, interactive interface that goes
beyond simple prediction by offering model consensus analysis, risk tier classification,
and uncertainty detection. The bulk CSV upload feature makes the tool practical for
real clinical workflows.

The project serves both as a functional clinical decision-support prototype and as an
educational resource demonstrating the mathematics behind modern machine learning
algorithms.

---

*Report generated for MedHelp v1.0 — Multiple Disease Recognition System*
*Built with Python 3.11 · NumPy · Streamlit*
