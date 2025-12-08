import streamlit as st
import pandas as pd
import pickle

# Load artifacts
preprocessor = pickle.load(open("preprocessor.pkl", "rb"))
model = pickle.load(open("pcos_model.pkl", "rb"))

st.title("PCOS Early Detection App")

st.write("Upload patient data for prediction:")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    X_p = preprocessor.transform(df)
    preds = model.predict(X_p)
    probs = model.predict_proba(X_p)[:,1]

    df["pcos_pred"] = preds
    df["pcos_prob"] = probs

    st.write(df)

    csv = df.to_csv(index=False)
    st.download_button("Download predictions", csv, "pcos_predictions.csv")
else:
    st.info("Upload a CSV file to begin.")
