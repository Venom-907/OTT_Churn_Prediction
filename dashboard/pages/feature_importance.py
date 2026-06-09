import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# PATH CONFIGURATION
# -----------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = ROOT_DIR / "models" / "trained_model.pkl"
FEATURES_PATH = ROOT_DIR / "models" / "feature_columns.pkl"

# -----------------------------
# PAGE TITLE
# -----------------------------
st.title("🌳 Feature Importance")

st.markdown("""
This page shows the most important features used by the
Random Forest model to predict customer churn.
""")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load(MODEL_PATH)

# -----------------------------
# LOAD FEATURE NAMES
# -----------------------------
feature_columns = joblib.load(FEATURES_PATH)

# -----------------------------
# FEATURE IMPORTANCE DATAFRAME
# -----------------------------
importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

top_features = importance_df.head(10)

# -----------------------------
# BAR CHART
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 4))

ax.barh(
    top_features["Feature"],
    top_features["Importance"]
)

ax.invert_yaxis()

ax.set_xlabel("Importance Score")
ax.set_ylabel("Features")
ax.set_title("Top 10 Feature Importances")

st.pyplot(fig)

# -----------------------------
# TABLE
# -----------------------------
st.subheader("📋 Feature Importance Table")

st.dataframe(
    top_features,
    use_container_width=True
)

# -----------------------------
# INSIGHTS
# -----------------------------
st.success("""
### Key Insights

• Average Watch Time Per Day is the most important feature.

• Engagement Score strongly impacts churn prediction.

• Watch Hours and Last Login Days are major behavioral indicators.

• Customers with low engagement are much more likely to churn.

• Subscription and payment behavior also influence churn risk.
""")