import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="Fraud Detection System", layout="wide")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("data/creditcard.csv")

model = load_model()
df = load_data()

st.title("💳 Credit Card Fraud Detection Dashboard")

# ================== GRAPH SECTION ==================

st.header("📊 Data Analysis")

col1, col2 = st.columns(2)

# Fraud vs Legit
with col1:
    st.subheader("Fraud vs Legit Transactions")
    fig1, ax1 = plt.subplots()
    df['Class'].value_counts().plot(kind='bar', ax=ax1)
    st.pyplot(fig1)

# Amount distribution
with col2:
    st.subheader("Transaction Amount Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(df['Amount'], bins=50, ax=ax2)
    st.pyplot(fig2)

# Heatmap
st.subheader("Correlation Heatmap")
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), cmap="coolwarm", ax=ax3)
st.pyplot(fig3)

# ================== FEATURE IMPORTANCE ==================

st.header("📌 Feature Importance")

importances = model.feature_importances_

fig4, ax4 = plt.subplots()
ax4.bar(range(len(importances)), importances)
ax4.set_title("Feature Importance")
st.pyplot(fig4)

# ================== SINGLE PREDICTION ==================

st.header("🔍 Predict Single Transaction")

features = []
cols = st.columns(3)

for i in range(30):
    with cols[i % 3]:
        val = st.slider(f"V{i}", -10.0, 10.0, 0.0)
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