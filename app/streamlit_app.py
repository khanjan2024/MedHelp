"""
MedHelp — Streamlit Web Application
=====================================
Multi-disease prediction dashboard using three scratch-built ML models.

Run:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

# Make project root importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.visualization import radar_chart, model_comparison_bar, loss_curve

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MedHelp",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS_DIR = os.path.join(ROOT, "models")

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stApp { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1a1d23, #252830);
        border-radius: 12px; padding: 20px; margin: 8px 0;
        border-left: 4px solid #636EFA;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .positive-result {
        background: linear-gradient(135deg, #3d0000, #5c0000);
        border-left: 4px solid #EF553B;
        border-radius: 12px; padding: 20px; margin: 10px 0;
        font-size: 1.1em; font-weight: bold; color: #ff6b6b;
    }
    .negative-result {
        background: linear-gradient(135deg, #003d1a, #005c28);
        border-left: 4px solid #00CC96;
        border-radius: 12px; padding: 20px; margin: 10px 0;
        font-size: 1.1em; font-weight: bold; color: #00e6a8;
    }
    .prob-badge {
        display: inline-block; padding: 6px 14px;
        border-radius: 20px; font-weight: bold; font-size: 1.1em;
        margin: 4px;
    }
    .section-header {
        font-size: 1.4em; font-weight: 700;
        color: #636EFA; margin: 20px 0 10px 0;
        border-bottom: 2px solid #636EFA; padding-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model loader (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(disease: str, model_type: str):
    """Load a pickled model from /models. Returns None if not found."""
    key = disease.replace(" ", "_")
    path = os.path.join(MODELS_DIR, f"{key}_{model_type}.sav")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_scaler(disease: str):
    key = disease.replace(" ", "_")
    path = os.path.join(MODELS_DIR, f"{key}_scaler.sav")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_features(disease: str):
    key = disease.replace(" ", "_")
    path = os.path.join(MODELS_DIR, f"{key}_features.sav")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_metrics(disease: str):
    key = disease.replace(" ", "_")
    path = os.path.join(MODELS_DIR, f"{key}_metrics.sav")
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


def models_trained(disease: str) -> bool:
    key = disease.replace(" ", "_")
    return os.path.exists(os.path.join(MODELS_DIR, f"{key}_logistic_regression.sav"))


# ---------------------------------------------------------------------------
# Prediction runner
# ---------------------------------------------------------------------------
def run_predictions(disease: str, input_array: np.ndarray):
    """
    Run input through all three models.
    Returns dict: { model_name: (prediction, probability) }
    """
    results = {}
    model_keys = {
        "Logistic Regression": "logistic_regression",
        "SVM": "svm",
        "Neural Network": "neural_network",
    }
    for display_name, file_key in model_keys.items():
        model = load_model(disease, file_key)
        if model is None:
            results[display_name] = (None, None)
            continue
        pred = int(model.predict(input_array)[0])
        prob = float(model.predict_proba(input_array)[0])
        results[display_name] = (pred, prob)
    return results


def align_and_scale(
    input_array: np.ndarray,
    form_values: dict,
    feature_cols: list,
    scaler,
) -> np.ndarray:
    """
    Build a full feature vector that matches the scaler's expected shape.

    - For features the form collected: use the form value.
    - For features the form did NOT collect (extra encoded columns from the
      CSV): fill with the scaler's mean (equivalent to 0 after z-scoring),
      which is the safest neutral imputation.
    """
    n_expected = len(feature_cols)
    full_vector = np.zeros((1, n_expected), dtype=float)
    form_lower = {k.lower(): v for k, v in form_values.items()}
    for i, col in enumerate(feature_cols):
        col_lower = col.lower()
        if col_lower in form_lower:
            full_vector[0, i] = float(form_lower[col_lower])
        else:
            full_vector[0, i] = float(scaler.mean_[i])
    return scaler.transform(full_vector)


def render_prediction_dashboard(results: dict, disease: str):
    """Render the side-by-side model comparison dashboard."""
    st.markdown('<div class="section-header">🔬 Model Comparison Dashboard</div>',
                unsafe_allow_html=True)

    cols = st.columns(3)
    model_probs = {}
    model_preds = {}

    for idx, (model_name, (pred, prob)) in enumerate(results.items()):
        with cols[idx]:
            st.markdown(f"**{model_name}**")
            if pred is None:
                st.warning("Model not trained yet.")
                continue

            model_probs[model_name] = prob
            model_preds[model_name] = pred
            label = "⚠️ Positive" if pred == 1 else "✅ Negative"
            css_class = "positive-result" if pred == 1 else "negative-result"
            prob_pct = round(prob * 100, 1)

            st.markdown(
                f'<div class="{css_class}">'
                f'{label}<br>'
                f'<span style="font-size:0.9em;">Risk Probability: <b>{prob_pct}%</b></span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.progress(min(int(prob_pct), 100))

    if model_probs:
        st.plotly_chart(model_comparison_bar(model_probs), use_container_width=True)

    if len(model_preds) >= 2:
        _render_consensus_panel(model_preds, model_probs)


def _render_consensus_panel(model_preds: dict, model_probs: dict):
    preds     = list(model_preds.values())
    probs     = list(model_probs.values())
    n_positive = sum(preds)
    n_total    = len(preds)
    avg_prob   = sum(probs) / n_total
    spread     = max(abs(p - avg_prob) for p in probs)
    unanimous  = (n_positive == 0 or n_positive == n_total)

    if not unanimous:
        tier, tier_color, tier_icon = "Uncertain",     "#f0a500", "⚠️"
    elif avg_prob < 0.30:
        tier, tier_color, tier_icon = "Low Risk",      "#00CC96", "🟢"
    elif avg_prob < 0.55:
        tier, tier_color, tier_icon = "Moderate Risk", "#f0a500", "🟡"
    else:
        tier, tier_color, tier_icon = "High Risk",     "#EF553B", "🔴"

    st.markdown("---")
    st.markdown('<div class="section-header">🧭 Consensus Analysis</div>',
                unsafe_allow_html=True)

    left, mid, right = st.columns(3)

    with left:
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#1a1d23,#252830);'
            f'border-radius:12px;padding:20px;text-align:center;'
            f'border:2px solid {tier_color};box-shadow:0 0 18px {tier_color}55;">'
            f'<div style="font-size:2.2em;">{tier_icon}</div>'
            f'<div style="font-size:1.3em;font-weight:700;color:{tier_color};margin-top:6px;">{tier}</div>'
            f'<div style="color:#aaa;font-size:0.85em;margin-top:4px;">Overall Risk Tier</div>'
            f'</div>', unsafe_allow_html=True)

    with mid:
        agreement_pct = int((max(n_positive, n_total - n_positive) / n_total) * 100)
        agree_color   = "#00CC96" if unanimous else "#f0a500"
        agree_label   = "Full Agreement" if unanimous else "Models Disagree"
        votes_str     = f"{n_positive} Positive · {n_total - n_positive} Negative"
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#1a1d23,#252830);'
            f'border-radius:12px;padding:20px;text-align:center;'
            f'border:2px solid {agree_color};box-shadow:0 0 18px {agree_color}55;">'
            f'<div style="font-size:2.2em;">{"🤝" if unanimous else "🔀"}</div>'
            f'<div style="font-size:1.3em;font-weight:700;color:{agree_color};margin-top:6px;">{agree_label}</div>'
            f'<div style="color:#aaa;font-size:0.85em;margin-top:4px;">{votes_str} &nbsp;|&nbsp; {agreement_pct}% consensus</div>'
            f'</div>', unsafe_allow_html=True)

    with right:
        spread_pct   = round(spread * 100, 1)
        spread_color = "#636EFA" if spread_pct < 15 else "#f0a500" if spread_pct < 30 else "#EF553B"
        spread_label = "Tight" if spread_pct < 15 else "Moderate" if spread_pct < 30 else "Wide"
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#1a1d23,#252830);'
            f'border-radius:12px;padding:20px;text-align:center;'
            f'border:2px solid {spread_color};box-shadow:0 0 18px {spread_color}55;">'
            f'<div style="font-size:2.2em;">📊</div>'
            f'<div style="font-size:1.3em;font-weight:700;color:{spread_color};margin-top:6px;">±{spread_pct}% Spread</div>'
            f'<div style="color:#aaa;font-size:0.85em;margin-top:4px;">Confidence Spread · {spread_label}</div>'
            f'</div>', unsafe_allow_html=True)

    if not unanimous:
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#2d2000,#3d2e00);'
            f'border-left:4px solid #f0a500;border-radius:10px;padding:16px;margin-top:16px;">'
            f'<b style="color:#f0a500;">⚠️ Uncertain Case Detected</b><br>'
            f'<span style="color:#e0c97f;font-size:0.95em;">'
            f'The models did not reach a unanimous verdict '
            f'({n_positive} predicted <b>Positive</b>, {n_total - n_positive} predicted <b>Negative</b>). '
            f'A confidence spread of <b>±{spread_pct}%</b> suggests the input sits near a decision boundary. '
            f'Consider consulting a medical professional for a definitive assessment.'
            f'</span></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Disease input forms
# ---------------------------------------------------------------------------
def diabetes_form():
    st.markdown('<div class="section-header">🩸 Diabetes Risk Assessment</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        pregnancies    = st.number_input("Pregnancies", 0, 20, 1)
        glucose        = st.number_input("Glucose (mg/dL)", 0, 300, 110)
        blood_pressure = st.number_input("Blood Pressure (mmHg)", 0, 200, 72)
    with c2:
        skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)
        insulin        = st.number_input("Insulin (μU/mL)", 0, 900, 80)
        bmi            = st.number_input("BMI", 0.0, 70.0, 25.0, step=0.1)
    with c3:
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.47, step=0.01)
        age = st.number_input("Age", 1, 120, 30)

    form_values = {
        "Pregnancies": pregnancies, "Glucose": glucose,
        "BloodPressure": blood_pressure, "SkinThickness": skin_thickness,
        "Insulin": insulin, "BMI": bmi,
        "DiabetesPedigreeFunction": dpf, "Age": age,
    }
    input_array = np.array(list(form_values.values())).reshape(1, -1)
    return input_array, form_values.copy(), form_values


def heart_form():
    st.markdown('<div class="section-header">❤️ Heart Disease Risk Assessment</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        age      = st.number_input("Age", 1, 120, 50)
        sex      = st.selectbox("Sex", ["Male (1)", "Female (0)"])
        sex_val  = 1 if "Male" in sex else 0
        cp       = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure", 80, 220, 120)
    with c2:
        chol    = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
        fbs     = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
        restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate Achieved", 60, 250, 150)
    with c3:
        exang   = st.selectbox("Exercise Induced Angina", [0, 1])
        oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 1.0, step=0.1)
        slope   = st.selectbox("Slope of Peak ST Segment (0-2)", [0, 1, 2])
        ca      = st.selectbox("Major Vessels Coloured (0-4)", [0, 1, 2, 3, 4])
        thal    = st.selectbox("Thalassemia (0-3)", [0, 1, 2, 3])

    form_values = {
        "age": age, "sex": sex_val, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
        "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
    }
    patient_dict = {
        "Age": age, "RestingBP": trestbps, "Cholesterol": chol,
        "MaxHR": thalach, "Oldpeak": oldpeak,
        "FastingBS": fbs, "RestingECG": restecg, "ExerciseAngina": exang,
    }
    input_array = np.array(list(form_values.values())).reshape(1, -1)
    return input_array, patient_dict, form_values


def liver_form():
    st.markdown('<div class="section-header">🫀 Liver Disease Risk Assessment</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        age        = st.number_input("Age", 1, 120, 40)
        gender     = st.selectbox("Gender", ["Male (1)", "Female (0)"])
        gender_val = 1 if "Male" in gender else 0
        tb         = st.number_input("Total Bilirubin", 0.0, 80.0, 1.0, step=0.1)
    with c2:
        db  = st.number_input("Direct Bilirubin", 0.0, 20.0, 0.3, step=0.1)
        ap  = st.number_input("Alkaline Phosphotase", 0, 2200, 200)
        alt = st.number_input("Alamine Aminotransferase", 0, 2000, 35)
    with c3:
        ast = st.number_input("Aspartate Aminotransferase", 0, 5000, 40)
        tp  = st.number_input("Total Proteins", 0.0, 10.0, 6.8, step=0.1)
        alb = st.number_input("Albumin", 0.0, 6.0, 3.5, step=0.1)
        agr = st.number_input("Albumin/Globulin Ratio", 0.0, 3.0, 1.0, step=0.01)

    form_values = {
        "Age": age, "Gender": gender_val,
        "TotalBilirubin": tb, "DirectBilirubin": db,
        "AlkalinePhosphotase": ap, "AlamineAminotransferase": alt,
        "AspartateAminotransferase": ast, "TotalProteins": tp,
        "Albumin": alb, "AlbuminAndGlobulinRatio": agr,
    }
    patient_dict = {
        "Age": age, "TotalBilirubin": tb, "DirectBilirubin": db,
        "AlkalinePhosphotase": ap, "AlamineAminotransferase": alt,
        "AspartateAminotransferase": ast, "TotalProteins": tp, "Albumin": alb,
    }
    input_array = np.array(list(form_values.values())).reshape(1, -1)
    return input_array, patient_dict, form_values


def kidney_form():
    st.markdown('<div class="section-header">🫘 Chronic Kidney Disease Risk Assessment</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", 1, 120, 45)
        bp  = st.number_input("Blood Pressure (mmHg)", 50, 200, 80)
        sg  = st.number_input("Specific Gravity", 1.000, 1.030, 1.020, step=0.001, format="%.3f")
        al  = st.selectbox("Albumin (0-5)", [0, 1, 2, 3, 4, 5])
        su  = st.selectbox("Sugar (0-5)", [0, 1, 2, 3, 4, 5])
    with c2:
        rbc_val = 1 if "Normal" in st.selectbox("Red Blood Cells", ["Normal (1)", "Abnormal (0)"]) else 0
        pc_val  = 1 if "Normal" in st.selectbox("Pus Cell", ["Normal (1)", "Abnormal (0)"]) else 0
        pcc_val = 1 if "Present" in st.selectbox("Pus Cell Clumps", ["Not Present (0)", "Present (1)"]) else 0
        ba_val  = 1 if "Present" in st.selectbox("Bacteria", ["Not Present (0)", "Present (1)"]) else 0
        bgr     = st.number_input("Blood Glucose Random (mg/dL)", 50, 500, 120)
    with c3:
        bu   = st.number_input("Blood Urea (mg/dL)", 1, 400, 40)
        sc   = st.number_input("Serum Creatinine (mg/dL)", 0.0, 80.0, 1.2, step=0.1)
        sod  = st.number_input("Sodium (mEq/L)", 100, 170, 137)
        pot  = st.number_input("Potassium (mEq/L)", 2.0, 10.0, 4.5, step=0.1)
        hemo = st.number_input("Hemoglobin (g/dL)", 3.0, 20.0, 13.5, step=0.1)
        pcv  = st.number_input("Packed Cell Volume (%)", 10, 60, 41)
        wc   = st.number_input("WBC Count (cells/cumm)", 2000, 20000, 8000)
        rc   = st.number_input("RBC Count (millions/cmm)", 2.0, 8.0, 4.7, step=0.1)

    form_values = {
        "age": age, "bp": bp, "sg": sg, "al": al, "su": su,
        "rbc": rbc_val, "pc": pc_val, "pcc": pcc_val, "ba": ba_val,
        "bgr": bgr, "bu": bu, "sc": sc, "sod": sod, "pot": pot,
        "hemo": hemo, "pcv": pcv, "wc": wc, "rc": rc,
    }
    patient_dict = {
        "Age": age, "BloodPressure": bp, "SpecificGravity": sg,
        "Albumin": al, "Sugar": su, "BloodUrea": bu,
        "SerumCreatinine": sc, "Hemoglobin": hemo,
    }
    input_array = np.array(list(form_values.values())).reshape(1, -1)
    return input_array, patient_dict, form_values


# ---------------------------------------------------------------------------
# Bulk CSV upload
# ---------------------------------------------------------------------------
def bulk_upload_section(disease: str):
    st.markdown('<div class="section-header">📂 Bulk Patient CSV Upload</div>',
                unsafe_allow_html=True)
    st.info(
        "Upload a CSV file with patient records. "
        "Columns must match the feature order used during training. "
        "The app will run all three models and return a downloadable predictions file."
    )
    uploaded = st.file_uploader("Upload patient CSV", type=["csv"], key=f"bulk_{disease}")
    if uploaded is None:
        return

    df = pd.read_csv(uploaded)
    st.write(f"Loaded **{len(df)}** records with columns: {list(df.columns)}")

    scaler = load_scaler(disease)
    if scaler is None:
        st.error("Scaler not found. Please train the models first.")
        return

    try:
        X_scaled = scaler.transform(df.values.astype(float))
    except Exception as e:
        st.error(f"Could not process CSV: {e}")
        return

    result_df = df.copy()
    for col_name, file_key in [("LR_Prediction","logistic_regression"),
                                ("SVM_Prediction","svm"),
                                ("NN_Prediction","neural_network")]:
        model = load_model(disease, file_key)
        if model is None:
            result_df[col_name] = "N/A"
            result_df[col_name.replace("Prediction","Probability")] = "N/A"
        else:
            result_df[col_name] = model.predict(X_scaled)
            result_df[col_name.replace("Prediction","Probability")] = (
                np.round(model.predict_proba(X_scaled) * 100, 2).astype(str) + "%"
            )

    st.dataframe(result_df, use_container_width=True)
    st.download_button(
        "⬇️ Download Predictions CSV",
        result_df.to_csv(index=False).encode("utf-8"),
        f"{disease.replace(' ','_')}_predictions.csv",
        "text/csv",
    )


# ---------------------------------------------------------------------------
# Training metrics sidebar widget
# ---------------------------------------------------------------------------
def show_training_metrics(disease: str):
    metrics = load_metrics(disease)
    if not metrics:
        return
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📊 Training Accuracy**")
    for model_name, acc in metrics.items():
        st.sidebar.metric(model_name, f"{acc}%")


# ---------------------------------------------------------------------------
# Disease page renderer
# ---------------------------------------------------------------------------
DISEASE_FORMS = {
    "Diabetes":      diabetes_form,
    "Heart Disease": heart_form,
    "Liver Disease": liver_form,
    "Kidney Disease":kidney_form,
}
DISEASE_ICONS = {
    "Diabetes":      "🩸",
    "Heart Disease": "❤️",
    "Liver Disease": "🫀",
    "Kidney Disease":"🫘",
}


def render_disease_page(disease: str):
    icon = DISEASE_ICONS.get(disease, "🏥")
    st.title(f"{icon} {disease} Prediction")
    st.caption("Powered by three scratch-built ML models: Logistic Regression · SVM · Neural Network")

    if not models_trained(disease):
        st.warning(
            f"⚠️ Models for **{disease}** have not been trained yet. "
            "Please run `python train_models.py` from the project root first."
        )

    show_training_metrics(disease)

    form_fn = DISEASE_FORMS[disease]
    input_array, patient_dict, form_values = form_fn()

    scaler       = load_scaler(disease)
    feature_cols = load_features(disease)
    if scaler is not None and feature_cols is not None:
        input_scaled = align_and_scale(input_array, form_values, feature_cols, scaler)
    elif scaler is not None:
        try:
            input_scaled = scaler.transform(input_array)
        except Exception:
            input_scaled = input_array
    else:
        input_scaled = input_array

    st.markdown("---")
    predict_col, _ = st.columns([1, 3])
    with predict_col:
        predict_clicked = st.button("🔍 Run Prediction", type="primary",
                                    use_container_width=True)

    if predict_clicked:
        with st.spinner("Running models..."):
            results = run_predictions(disease, input_scaled)

        render_prediction_dashboard(results, disease)

        st.markdown("---")
        st.markdown('<div class="section-header">📡 Health Metrics Radar Chart</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(radar_chart(patient_dict, disease), use_container_width=True)

        st.markdown("---")
        st.markdown('<div class="section-header">📉 Training Loss Curves</div>',
                    unsafe_allow_html=True)
        lc1, lc2, lc3 = st.columns(3)
        for col, (mname, mkey) in zip([lc1, lc2, lc3], [
            ("Logistic Regression","logistic_regression"),
            ("SVM","svm"),
            ("Neural Network","neural_network"),
        ]):
            model = load_model(disease, mkey)
            if model and hasattr(model, "loss_history") and model.loss_history:
                with col:
                    st.plotly_chart(loss_curve(model.loss_history, mname),
                                    use_container_width=True)

    st.markdown("---")
    bulk_upload_section(disease)


# ---------------------------------------------------------------------------
# Home page
# ---------------------------------------------------------------------------
def render_home():
    st.title("🏥 MedHelp")
    st.subheader("Advanced Multiple Disease Recognition System")

    st.markdown("""
    Welcome to **MedHelp** — a clinical decision-support tool that uses three
    machine-learning models built entirely from scratch (no scikit-learn, no TensorFlow)
    to predict the risk of four major diseases.

    ---
    ### 🧠 Models
    | Model | Algorithm | Loss Function |
    |---|---|---|
    | Logistic Regression | Sigmoid + Gradient Descent | Binary Cross-Entropy |
    | SVM Classifier | Hyperplane + Hinge Loss | Hinge Loss |
    | Neural Network | Single Hidden Layer + Sigmoid | Cross-Entropy |

    ---
    ### 🗂️ Supported Diseases
    """)

    cols = st.columns(4)
    for col, (icon, name, dataset) in zip(cols, [
        ("🩸","Diabetes","Pima Indians Diabetes Dataset"),
        ("❤️","Heart Disease","UCI Heart Disease (Cleveland)"),
        ("🫀","Liver Disease","Indian Liver Patient Records"),
        ("🫘","Kidney Disease","Chronic Kidney Disease Dataset"),
    ]):
        trained = models_trained(name)
        status  = "✅ Trained" if trained else "⏳ Not trained"
        col.markdown(
            f'<div class="metric-card">'
            f'<div style="font-size:2em">{icon}</div>'
            f'<b>{name}</b><br>'
            f'<small style="color:#aaa">{dataset}</small><br>'
            f'<small>{status}</small>'
            f'</div>', unsafe_allow_html=True)

    st.markdown("""
    ---
    ### 🚀 Getting Started
    1. Place your Kaggle CSV files in the `/data` folder.
    2. Run `python train_models.py` to train and save all models.
    3. Use the sidebar to navigate to a disease and enter patient data.
    4. Hit **Run Prediction** to see results from all three models simultaneously.
    5. Use the **Bulk Upload** section to process multiple patients at once.

    > ⚠️ **Disclaimer:** This tool is for educational and research purposes only.
    > It is not a substitute for professional medical advice.
    """)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/caduceus.png", width=80)
        st.markdown("## MedHelp")
        st.caption("Multiple Disease Recognition System")
        st.markdown("---")

        selected = option_menu(
            menu_title=None,
            options=["Home","Diabetes","Heart Disease","Liver Disease","Kidney Disease"],
            icons=["house-fill","droplet-fill","heart-fill","activity","capsule"],
            default_index=0,
            styles={
                "container":        {"background-color": "#1a1d23"},
                "icon":             {"color": "#636EFA", "font-size": "16px"},
                "nav-link":         {"font-size": "14px", "color": "#c9d1d9",
                                     "--hover-color": "#252830"},
                "nav-link-selected":{"background-color": "#252830",
                                     "color": "#636EFA", "font-weight": "bold"},
            },
        )

        st.markdown("---")
        st.caption("Built with ❤️ using pure NumPy ML")

    if selected == "Home":
        render_home()
    else:
        render_disease_page(selected)


if __name__ == "__main__":
    main()
