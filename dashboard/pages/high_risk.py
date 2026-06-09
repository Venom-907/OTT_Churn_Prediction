import streamlit as st
import pandas as pd
import joblib

# -----------------------------------
# Page Title
# -----------------------------------

st.title("🚨 High Risk Customers")

st.markdown(
    "Customers most likely to churn based on the trained Random Forest model."
)

# -----------------------------------
# Load Data
# -----------------------------------

@st.cache_data
def load_raw_data():
    return pd.read_csv(
        "../data/processed/final_dataset.csv"
    )


@st.cache_data
def load_encoded_data():
    return pd.read_csv(
        "../data/processed/final_dataset_encoded.csv"
    )


raw_df = load_raw_data()
encoded_df = load_encoded_data()

# -----------------------------------
# Load Model
# -----------------------------------

model = joblib.load(
    "../models/trained_model.pkl"
)

# -----------------------------------
# Prepare Features
# -----------------------------------

X = encoded_df.drop(
    "churned",
    axis=1
)

# -----------------------------------
# Predict Churn Probability
# -----------------------------------

encoded_df["churn_probability"] = (
    model.predict_proba(X)[:, 1]
)

# Add customer IDs back
encoded_df["customer_id"] = raw_df["customer_id"]

# -----------------------------------
# Sort Customers by Risk
# -----------------------------------

high_risk_df = encoded_df.sort_values(
    by="churn_probability",
    ascending=False
)

top_10 = high_risk_df[
    ["customer_id", "churn_probability"]
].head(10)

# -----------------------------------
# KPI Cards
# -----------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Customers",
        len(raw_df)
    )

with col2:
    st.metric(
        "Customers Above 80% Risk",
        len(
            high_risk_df[
                high_risk_df["churn_probability"] > 0.80
            ]
        )
    )

with col3:
    st.metric(
        "Highest Risk",
        f"{high_risk_df['churn_probability'].max()*100:.2f}%"
    )

st.markdown("---")

# -----------------------------------
# Top 10 High Risk Customers
# -----------------------------------

st.subheader("🔥 Top 10 Highest Risk Customers")

st.warning(
    "These customers have the highest predicted probability of churn and should be targeted with retention campaigns."
)

display_df = top_10.copy()

display_df["churn_probability"] = (
    display_df["churn_probability"] * 100
).round(2)

display_df.rename(
    columns={
        "customer_id": "Customer ID",
        "churn_probability": "Churn Probability (%)"
    },
    inplace=True
)

st.dataframe(
    display_df,
    use_container_width=True
)

# -----------------------------------
# Download Button
# -----------------------------------

csv = display_df.to_csv(index=False)

st.download_button(
    label="📥 Download Top Risk Customers",
    data=csv,
    file_name="high_risk_customers.csv",
    mime="text/csv"
)