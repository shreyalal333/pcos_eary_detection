import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import sklearn

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="PCOS Early Detection",
    layout="wide"
)

st.title("🩺 PCOS Early Detection System")
st.caption("AI-based screening tool (Educational use only)")

# ---------------- LOAD MODEL ----------------
try:
    with open("preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    st.success("✅ Model & Preprocessor Loaded")

except Exception as e:
    st.error("❌ Failed to load model files")
    st.exception(e)
    st.stop()

# ---------------- FILE UPLOAD ----------------
st.header("📤 Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV file (must include PCOS column)",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # ---------------- LABEL CHECK ----------------
    if "PCOS" not in df.columns:
        st.error("❌ Dataset must contain a column named 'PCOS' (0 = No, 1 = Yes)")
        st.stop()

    y_true = df["SOPK"]
    X = df.drop(columns=["SOPK"])

    # ---------------- COLUMN CHECK ----------------
    expected_cols = list(preprocessor.feature_names_in_)

    missing = set(expected_cols) - set(X.columns)
    extra = set(X.columns) - set(expected_cols)

    if missing:
        st.error("❌ Missing columns: " + ", ".join(missing))
        st.stop()

    if extra:
        X = X[expected_cols]

    # ---------------- TRANSFORM & PREDICT ----------------
    X_p = preprocessor.transform(X)
    y_pred = model.predict(X_p)
    y_prob = model.predict_proba(X_p)[:, 1]

    # ---------------- RESULTS TABLE ----------------
    results = X.copy()
    results["Actual_PCOS"] = y_true.values
    results["Predicted_PCOS"] = y_pred
    results["PCOS_Probability"] = y_prob

    st.subheader("📊 Prediction Results")
    st.dataframe(results)

    # ================== EVALUATION ==================
    st.header("📈 Model Evaluation")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2f}")
    col2.metric("Precision", f"{precision_score(y_true, y_pred):.2f}")
    col3.metric("Recall", f"{recall_score(y_true, y_pred):.2f}")
    col4.metric("F1 Score", f"{f1_score(y_true, y_pred):.2f}")

    # ---------------- CONFUSION MATRIX ----------------
    st.subheader("🔁 Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)

    fig_cm, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No PCOS", "PCOS"]
    )
    disp.plot(ax=ax, cmap="Blues")
    st.pyplot(fig_cm)

    # ---------------- ROC CURVE ----------------
    st.subheader("📉 ROC Curve")

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig_roc, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig_roc)

    # ================== SHAP ==================
    st.header("🧠 Model Explainability (SHAP)")

    with st.spinner("Generating SHAP values..."):
        explainer = shap.Explainer(model)
        shap_values = explainer(X_p[:50])

        fig_shap, ax = plt.subplots()
        shap.plots.beeswarm(shap_values, show=False)
        st.pyplot(fig_shap)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("⚠️ Educational use only. Not a medical diagnosis.")
