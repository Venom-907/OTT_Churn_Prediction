import streamlit as st
import pandas as pd
from pathlib import Path

# -----------------------------
# PATH CONFIGURATION
# -----------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

DATA_PATH = ROOT_DIR / "data" / "processed" / "final_dataset.csv"

# -----------------------------
# PAGE TITLE
# -----------------------------
st.title("🔍 Customer Lookup")

st.markdown(
    "Search any customer and view churn risk, profile details, and retention recommendations."
)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# -----------------------------
# SEARCH BOX
# -----------------------------
customer_id = st.text_input(
    "Enter Customer ID",
    placeholder="Example: a9b75100-82a8-427a-a208-72f24052884a"
)

# -----------------------------
# SEARCH BUTTON
# -----------------------------
if st.button("🔎 Search Customer"):

    result = df[df["customer_id"] == customer_id]

    if result.empty:

        st.error("Customer not found!")

    else:

        customer = result.iloc[0]

        st.success("Customer Found")

        st.divider()

        # -----------------------------
        # CUSTOMER DETAILS
        # -----------------------------
        st.subheader("👤 Customer Profile")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Customer ID:**", customer["customer_id"])
            st.write("**Age:**", customer["age"])
            st.write("**Gender:**", customer["gender"])
            st.write("**Subscription:**", customer["subscription_type"])
            st.write("**Region:**", customer["region"])
            st.write("**Device:**", customer["device"])

        with col2:
            st.write("**Watch Hours:**", customer["watch_hours"])
            st.write("**Last Login Days:**", customer["last_login_days"])
            st.write("**Monthly Fee:**", customer["monthly_fee"])
            st.write("**Profiles:**", customer["number_of_profiles"])
            st.write("**Favorite Genre:**", customer["favorite_genre"])
            st.write(
                "**Engagement Score:**",
                round(customer["engagement_score"], 2)
            )

        st.divider()

        # -----------------------------
        # CHURN RESULT
        # -----------------------------
        churn = customer["churned"]

        st.subheader("🎯 Churn Status")

        if churn == 1:

            st.error("⚠ High Churn Risk Customer")

            st.metric(
                "Churn Probability",
                "100%"
            )

        else:

            st.success("✅ Customer Likely To Stay")

            st.metric(
                "Churn Probability",
                "0%"
            )

        st.divider()

        # -----------------------------
        # RECOMMENDATIONS
        # -----------------------------
        st.subheader("💡 Retention Recommendations")

        if churn == 1:

            st.warning("""
• Offer discount coupon

• Send personalized recommendations

• Free Premium Trial

• Re-engagement Email Campaign

• Special Retention Offer
""")

        else:

            st.success("""
• Maintain current experience

• Recommend premium upgrade

• Suggest new content

• Reward loyalty
""")