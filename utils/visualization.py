"""
Visualization utilities for MediPredictML.
Provides radar charts and bar charts comparing patient metrics
against healthy baselines using Plotly.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


# ---------------------------------------------------------------------------
# Healthy baseline reference values per disease
# ---------------------------------------------------------------------------
HEALTHY_BASELINES = {
    "Diabetes": {
        "Pregnancies": 1,
        "Glucose": 90,
        "BloodPressure": 70,
        "SkinThickness": 20,
        "Insulin": 80,
        "BMI": 22,
        "DiabetesPedigreeFunction": 0.3,
        "Age": 30,
    },
    "Heart Disease": {
        "Age": 45,
        "RestingBP": 120,
        "Cholesterol": 180,
        "MaxHR": 150,
        "Oldpeak": 0.5,
        "FastingBS": 0,
        "RestingECG": 0,
        "ExerciseAngina": 0,
    },
    "Liver Disease": {
        "Age": 35,
        "TotalBilirubin": 0.8,
        "DirectBilirubin": 0.2,
        "AlkalinePhosphotase": 90,
        "AlamineAminotransferase": 25,
        "AspartateAminotransferase": 25,
        "TotalProteins": 7.0,
        "Albumin": 4.0,
    },
    "Kidney Disease": {
        "Age": 40,
        "BloodPressure": 75,
        "SpecificGravity": 1.02,
        "Albumin": 0,
        "Sugar": 0,
        "BloodUrea": 30,
        "SerumCreatinine": 1.0,
        "Hemoglobin": 14,
    },
}


def radar_chart(patient_values: dict, disease: str) -> go.Figure:
    """
    Generate a radar (spider) chart comparing patient values
    against healthy baseline values.

    Parameters
    ----------
    patient_values : dict
        Feature name → patient's input value.
    disease : str
        One of the keys in HEALTHY_BASELINES.

    Returns
    -------
    plotly Figure
    """
    baseline = HEALTHY_BASELINES.get(disease, {})
    # Only keep features present in both dicts
    features = [f for f in baseline if f in patient_values]
    if not features:
        return go.Figure()

    patient_vals = [float(patient_values[f]) for f in features]
    baseline_vals = [float(baseline[f]) for f in features]

    # Normalise both to [0, 1] relative to max(patient, baseline) per feature
    max_vals = [max(p, b, 1e-9) for p, b in zip(patient_vals, baseline_vals)]
    patient_norm = [p / m for p, m in zip(patient_vals, max_vals)]
    baseline_norm = [b / m for b, m in zip(baseline_vals, max_vals)]

    # Close the polygon
    features_closed = features + [features[0]]
    patient_norm_closed = patient_norm + [patient_norm[0]]
    baseline_norm_closed = baseline_norm + [baseline_norm[0]]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=patient_norm_closed,
            theta=features_closed,
            fill="toself",
            name="Your Values",
            line_color="#EF553B",
            fillcolor="rgba(239,85,59,0.2)",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=baseline_norm_closed,
            theta=features_closed,
            fill="toself",
            name="Healthy Baseline",
            line_color="#00CC96",
            fillcolor="rgba(0,204,150,0.2)",
        )
    )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title=f"{disease} — Patient vs Healthy Baseline",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white"),
        legend=dict(bgcolor="#1a1d23"),
    )
    return fig


def model_comparison_bar(results: dict) -> go.Figure:
    """
    Bar chart comparing probability scores from all three models.

    Parameters
    ----------
    results : dict
        { model_name: probability_float (0-1) }

    Returns
    -------
    plotly Figure
    """
    models = list(results.keys())
    probs = [round(v * 100, 2) for v in results.values()]
    colors = ["#636EFA", "#EF553B", "#00CC96"]

    fig = go.Figure(
        go.Bar(
            x=models,
            y=probs,
            marker_color=colors[: len(models)],
            text=[f"{p}%" for p in probs],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Model Probability Comparison",
        yaxis=dict(title="Risk Probability (%)", range=[0, 110]),
        xaxis=dict(title="Model"),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
        font=dict(color="white"),
        showlegend=False,
    )
    return fig


def feature_importance_bar(weights: np.ndarray, feature_names: list[str]) -> go.Figure:
    """
    Horizontal bar chart of absolute logistic regression weights
    as a proxy for feature importance.
    """
    importance = np.abs(weights)
    sorted_idx = np.argsort(importance)
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importance = importance[sorted_idx]

    fig = go.Figure(
        go.Bar(
            x=sorted_importance,
            y=sorted_features,
            orientation="h",
            marker_color="#636EFA",
        )
    )
    fig.update_layout(
        title="Feature Importance (|LR Weights|)",
        xaxis_title="Absolute Weight",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
        font=dict(color="white"),
    )
    return fig


def loss_curve(loss_history: list[float], model_name: str) -> go.Figure:
    """Line chart of training loss over iterations/epochs."""
    fig = px.line(
        x=list(range(len(loss_history))),
        y=loss_history,
        labels={"x": "Iteration / Epoch", "y": "Loss"},
        title=f"{model_name} — Training Loss Curve",
    )
    fig.update_layout(
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
        font=dict(color="white"),
    )
    return fig
