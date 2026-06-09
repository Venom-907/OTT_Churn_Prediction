import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ----------------------------------
# PATH CONFIGURATION
# ----------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = ROOT_DIR / "models" / "trained_model.pkl"
FEATURES_PATH = ROOT_DIR / "models" / "feature_columns.pkl"

# ----------------------------------
# LOAD MODEL & FEATURE COLUMNS
# ----------------------------------

model = joblib.load(MODEL_PATH)

feature_columns = joblib.load(FEATURES_PATH)

# ----------------------------------
# PAGE TITLE
# ----------------------------------

st.title("🤖 Customer Churn Prediction")

st.markdown(
    "Enter customer information and predict churn risk."
)

# ----------------------------------
# INPUT LAYOUT
# ----------------------------------

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female", "Other"]
    )

    subscription_type = st.selectbox(
        "Subscription Type",
        ["Basic", "Standard", "Premium"]
    )

    region = st.selectbox(
        "Region",
        [
            "Africa",
            "Asia",
            "Europe",
            "North America",
            "South America",
            "Oceania"
        ]
    )

    device = st.selectbox(
        "Device",
        [
            "Mobile",
            "TV",
            "Laptop",
            "Desktop",
            "Tablet"
        ]
    )

with col2:

    watch_hours = st.number_input(
        "Watch Hours",
        min_value=0.0,
        value=10.0
    )

    last_login_days = st.number_input(
        "Last Login Days",
        min_value=0,
        value=5
    )

    monthly_fee = st.number_input(
        "Monthly Fee",
        min_value=0.0,
        value=13.99
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Credit Card",
            "Debit Card",
            "PayPal",
            "Crypto",
            "Gift Card"
        ]
    )

    number_of_profiles = st.number_input(
        "Number of Profiles",
        min_value=1,
        max_value=10,
        value=2
    )

avg_watch_time_per_day = st.number_input(
    "Average Watch Time Per Day",
    min_value=0.0,
    value=1.0
)

favorite_genre = st.selectbox(
    "Favorite Genre",
    [
        "Action",
        "Drama",
        "Comedy",
        "Sci-Fi",
        "Romance",
        "Horror",
        "Documentary"
    ]
)

# ----------------------------------
# PREDICTION
# ----------------------------------

if st.button("🔮 Predict Churn"):

    engagement_score = (
        watch_hours
        + avg_watch_time_per_day
        - last_login_days
    )

    input_data = {
        "age": age,
        "watch_hours": watch_hours,
        "last_login_days": last_login_days,
        "monthly_fee": monthly_fee,
        "number_of_profiles": number_of_profiles,
        "avg_watch_time_per_day": avg_watch_time_per_day,
        "engagement_score": engagement_score
    }

    input_df = pd.DataFrame([input_data])

    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    mapping = {
        f"gender_{gender}": 1,
        f"subscription_type_{subscription_type}": 1,
        f"region_{region}": 1,
        f"device_{device}": 1,
        f"payment_method_{payment_method}": 1,
        f"favorite_genre_{favorite_genre}": 1
    }

    for col, value in mapping.items():
        if col in input_df.columns:
            input_df[col] = value

    input_df = input_df[feature_columns]

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    st.markdown("---")

    st.write("Raw Probability:", probability)

    if prediction == 1:
        st.error("⚠️ Customer Likely To Churn")
    else:
        st.success("✅ Customer Likely To Stay")

    st.metric(
        "Churn Probability",
        f"{probability * 100:.2f}%"
    )

    if probability >= 0.80:
        st.error("🔴 HIGH RISK")

    elif probability >= 0.50:
        st.warning("🟡 MEDIUM RISK")

    else:
        st.success("🟢 LOW RISK")