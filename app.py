import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Page config
st.set_page_config(page_title="Fraud Detection System", layout="wide")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("💳 Credit Card Fraud Detection System")

st.markdown("### Enter transaction features below")

# ================== SINGLE PREDICTION ==================

features = []
cols = st.columns(3)

for i in range(30):
    with cols[i % 3]:
        val = st.slider(f"Feature V{i}", -10.0, 10.0, 0.0)
        features.append(val)

if st.button("🚀 Predict Fraud"):
    data = np.array(features).reshape(1, -1)

    prediction = model.predict(data)
    probability = model.predict_proba(data)

    fraud_prob = probability[0][1] * 100

    if prediction[0] == 1:
        st.error("⚠️ Fraudulent Transaction")
    else:
        st.success("✅ Legit Transaction")

    st.metric("Fraud Probability", f"{fraud_prob:.2f}%")
    st.progress(int(fraud_prob))


# ================== FEATURE IMPORTANCE ==================

st.header("📌 Feature Importance")

if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_

    fig_data = pd.DataFrame({
        "Feature": [f"V{i}" for i in range(len(importances))],
        "Importance": importances
    })

    st.bar_chart(fig_data.set_index("Feature"))
else:
    st.info("Feature importance not available for this model.")


# ================== CSV UPLOAD ==================

st.header("📂 Upload CSV for Bulk Prediction")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write("Preview of Uploaded Data:")
    st.write(data.head())

    if st.button("Run Bulk Prediction"):
        predictions = model.predict(data)
        data["Prediction"] = predictions

        st.write("Results:")
        st.write(data.head())

        st.download_button(
            label="Download Results",
            data=data.to_csv(index=False),
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )