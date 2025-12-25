%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn
import shap
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="PCOS Early Detection",
    layout="wide",
)

st.title("🩺 PCOS Early Detection System")
st.caption("AI-based screening tool (Educational use only)")

# ---------------- VERSION CHECK ----------------
with st.expander("🔍 Environment Info"):
    st.write({
        "NumPy": np.__version__,
        "Pandas": pd.__version__,
        "Scikit-learn": sklearn.__version__,
    })

# ---------------- LOAD MODEL ----------------
try:
    with open("preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    st.success("✅ Model & preprocessor loaded successfully")

except Exception as e:
    st.error("❌ Failed to load model files")
    st.exception(e)
    st.stop()

# ---------------- FILE UPLOAD ----------------
st.header("📤 Upload Patient Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV file with patient data",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    # ---------------- COLUMN CHECK ----------------
    expected_cols = list(preprocessor.feature_names_in_)
    uploaded_cols = df.columns.tolist()

    missing = set(expected_cols) - set(uploaded_cols)
    extra = set(uploaded_cols) - set(expected_cols)

    if missing:
        st.error("❌ Missing columns: " + ", ".join(missing))
        st.stop()

    if extra:
        df = df[expected_cols]

    # ---------------- TRANSFORM & PREDICT ----------------
    X_processed = preprocessor.transform(df)
    preds = model.predict(X_processed)
    probs = model.predict_proba(X_processed)[:, 1]

    results = df.copy()
    results["PCOS_Prediction"] = preds
    results["PCOS_Risk_Probability"] = probs

    st.subheader("📊 Prediction Results")
    st.dataframe(results)

    # ---------------- SHAP EXPLANATION ----------------
    st.subheader("🧠 Model Explanation (SHAP)")

    with st.spinner("Generating SHAP explanation..."):
        explainer = shap.Explainer(model)
        shap_values = explainer(X_processed[:50])

        fig, ax = plt.subplots()
        shap.plots.beeswarm(shap_values, show=False)
        st.pyplot(fig)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("⚠️ This tool is for academic demonstration only. Not a medical diagnosis.")
