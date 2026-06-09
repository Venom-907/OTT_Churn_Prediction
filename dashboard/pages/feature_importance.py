import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Page Title
st.title("🌳 Feature Importance")

st.markdown("""
This page shows the most important features used by the
Random Forest model to predict customer churn.
""")

# Load Model
model = joblib.load("../models/trained_model.pkl")

# Load Feature Names
feature_columns = joblib.load("../models/feature_columns.pkl")

# Create Feature Importance DataFrame
importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": model.feature_importances_
})

# Sort Features
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

# Top 10 Features
top_features = importance_df.head(10)

# Chart
fig, ax = plt.subplots(figsize=(10, 6))

ax.barh(
    top_features["Feature"],
    top_features["Importance"]
)

ax.invert_yaxis()

ax.set_xlabel("Importance Score")
ax.set_ylabel("Features")
ax.set_title("Top 10 Feature Importances")

st.pyplot(fig)

# Display Table
st.subheader("📋 Feature Importance Table")

st.dataframe(
    top_features,
    use_container_width=True
)

# Insights
st.success("""
### Key Insights

• Average Watch Time Per Day is the most important feature.

• Engagement Score strongly impacts churn prediction.

• Watch Hours and Last Login Days are major behavioral indicators.

• Customers with low engagement are much more likely to churn.

• Subscription and payment behavior also influence churn risk.
""")