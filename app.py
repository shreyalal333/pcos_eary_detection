import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="PCOS Detection with Explainability", layout="wide")
st.title("🩺 PCOS Early Detection System with Explainable AI")
st.write("Upload patient data to get predictions and understand *why* the model predicts PCOS.")

# ------------------------------------------------------
# LOAD MODEL + PREPROCESSOR
# ------------------------------------------------------
try:
    with open("preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    with open("pcos_model.pkl", "rb") as f:
        model = pickle.load(f)
    st.success("✔ Model loaded successfully")
except Exception as e:
    st.error("❌ Could not load model files. Ensure `.pkl` files exist.")
    st.stop()


# ------------------------------------------------------
# MAIN UI TABS
# ------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📂 Batch Prediction", "🎯 Single Patient Input", "🧠 Model Explainability (SHAP)"])


# ------------------------------------------------------
# TAB 1 — BATCH PREDICTION
# ------------------------------------------------------
with tab1:
    st.header("📂 Upload CSV for Batch Predictions")

    uploaded = st.file_uploader("Upload patient dataset (CSV)", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.subheader("📌 Uploaded Data Preview")
        st.write(df.head())

        # Validate Columns
        expected = preprocessor.feature_names_in_.tolist()
        uploaded_cols = df.columns.tolist()

        missing = set(expected) - set(uploaded_cols)
        extra = set(uploaded_cols) - set(expected)

        if missing:
            st.error(f"❌ Missing columns: {', '.join(missing)}")
            st.stop()

        if extra:
            st.warning(f"⚠ Extra columns ignored: {', '.join(extra)}")
            df = df[expected]

        # Prediction
        X_p = preprocessor.transform(df)
        preds = model.predict(X_p)
        probs = model.predict_proba(X_p)[:, 1]

        df["PCOS_prediction"] = preds
        df["PCOS_probability"] = probs

        st.success("✔ Predictions completed")
        st.write(df)

        csv = df.to_csv(index=False)
        st.download_button("📥 Download Predictions", csv, "pcos_predictions.csv")


# ------------------------------------------------------
# TAB 2 — SINGLE PATIENT INPUT
# ------------------------------------------------------
with tab2:
    st.header("🎯 Enter Patient Details Manually")

    expected_cols = preprocessor.feature_names_in_.tolist()
    user_input = {}

    # Automatically generate number/text inputs for expected columns
    for col in expected_cols:
        user_input[col] = st.number_input(f"{col}", value=0.0)

    if st.button("Predict for this patient"):
        df_single = pd.DataFrame([user_input])
        X_p = preprocessor.transform(df_single)

        pred = model.predict(X_p)[0]
        prob = model.predict_proba(X_p)[0][1]

        st.subheader("Prediction Result")
        if pred == 1:
            st.error(f"🔴 High PCOS Probability: **{prob:.2f}**")
        else:
            st.success(f"🟢 Low PCOS Probability: **{prob:.2f}**")

        st.info("Go to the SHAP tab to understand why this result occurred.")


# ------------------------------------------------------
# TAB 3 — SHAP EXPLAINABILITY
# ------------------------------------------------------
with tab3:
    st.header("🧠 Model Explainability Using SHAP")

    st.write("""
    SHAP (SHapley Additive exPlanations) helps visualize how each feature  
    contributes to the prediction. This increases transparency and trust in the model.
    """)

    sample_size = st.slider("Select number of samples to explain", 10, 200, 50)

    # SHAP processing
    try:
        explainer = shap.TreeExplainer(model)

        # Create background data
        X_background = pd.DataFrame(
            preprocessor.transform(pd.DataFrame(
                np.random.rand(sample_size, len(preprocessor.feature_names_in_)), 
                columns=preprocessor.feature_names_in_
            ))
        )

        shap_values = explainer.shap_values(X_background)

        st.subheader("📊 SHAP Summary Plot")
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_background, feature_names=preprocessor.feature_names_in_, show=False)
        st.pyplot(fig)

        st.subheader("🎨 SHAP Force Plot (Single Instance)")
        index = st.slider("Choose a sample index", 0, sample_size - 1, 0)

        force_plot_html = shap.plots.force(
            explainer.expected_value,
            shap_values[index],
            X_background.iloc[index, :],
            matplotlib=False
        )
        st.components.v1.html(force_plot_html.html(), height=300)

    except Exception as e:
        st.error("❌ SHAP could not be generated. Ensure the model is tree-based like XGBoost/RandomForest.")
        st.write(str(e))
