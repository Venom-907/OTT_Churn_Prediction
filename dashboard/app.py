import streamlit as st
import pandas as pd
import joblib


st.markdown("""
<style>

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


#Page Configuration
st.set_page_config(
    page_title="OTT Churn Prediction System",
    page_icon="🎬",
    layout="wide"
)

#Load Dataset
@st.cache_data
def load_data():
    return pd.read_csv("../data/processed/final_dataset.csv")

df = load_data()

#Sidebar
st.sidebar.markdown("""
# 🎬 OTT Churn Prediction

### Customer Retention Analytics

---
Machine Learning Dashboard

**Model:** Random Forest

**Accuracy:** 97.5%

---
""")

#Main Title
st.title("🎬 OTT Subscriber Churn Prediction & Retention Analytics System")

st.markdown("---")

st.info("""
Predict customer churn and identify at-risk subscribers using Machine Learning.

This dashboard helps OTT platforms improve customer retention through behavioral analytics and predictive modeling.
""")


#KPI Cards
total_customers = len(df)

churn_rate = round(
    df["churned"].mean() * 100,
    2
)

avg_watch_hours = round(
    df["watch_hours"].mean(),
    2
)

avg_engagement = round(
    df["engagement_score"].mean(),
    2
)

#Display them
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Customers",
        total_customers
    )

with col2:
    st.metric(
        "Churn Rate",
        f"{churn_rate}%"
    )

with col3:
    st.metric(
        "Avg Watch Hours",
        avg_watch_hours
    )

with col4:
    st.metric(
        "Avg Engagement",
        avg_engagement
    )
    
#Project Overview
st.markdown("---")

st.header("📖 Project Overview")

st.write(
    """
    This project predicts customer churn in an OTT platform using Machine Learning.

    The system analyzes customer behavior, engagement patterns,
    subscription details, and viewing activity to identify
    users who are likely to churn.

    Key objective:
    Improve customer retention through predictive analytics.
    """
)


st.subheader("🔄 System Workflow")

st.markdown("""
1. Customer behavioral data collected

2. Data preprocessing and feature engineering

3. Random Forest model training

4. Churn prediction generation

5. High-risk customer identification

6. Retention strategy recommendations
""")

#Model Performance Table
st.markdown("---")

st.header("🤖 Model Performance")

#Create Dataframe
performance_df = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        "88.7%",
        "97.5%",
        "97.5%"
    ]
})

st.dataframe(
    performance_df,
    use_container_width=True
)

#Best Model
st.success(
    "🏆 Random Forest selected as final model with 97.5% accuracy."
)