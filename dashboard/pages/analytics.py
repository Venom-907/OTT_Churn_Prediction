import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# =====================================
# PAGE TITLE
# =====================================

st.title("📊 Analytics Dashboard")
st.markdown("Customer behavior and churn analysis.")

# =====================================
# LOAD DATA
# =====================================

@st.cache_data
def load_data():
    return pd.read_csv("../data/processed/final_dataset.csv")

df = load_data()

# =====================================
# KPI CARDS
# =====================================

total_customers = len(df)
churn_rate = round(df["churned"].mean() * 100, 2)
avg_watch = round(df["watch_hours"].mean(), 2)
avg_engagement = round(df["engagement_score"].mean(), 2)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Customers", total_customers)
c2.metric("Churn Rate", f"{churn_rate}%")
c3.metric("Avg Watch Hours", avg_watch)
c4.metric("Avg Engagement", avg_engagement)

st.divider()

st.subheader("📊 Business Insights")

# =====================================
# SMALL CUSTOMER DISTRIBUTION
# =====================================

col1, col2 = st.columns([1, 1])

with col1:

    st.subheader("Customer Distribution")

    churn_counts = df["churned"].value_counts()

    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    ax.pie(
        churn_counts,
        labels=["Stayed", "Churned"],
        autopct="%1.1f%%",
        startangle=90
    )

    plt.tight_layout()

    st.pyplot(fig)

with col2:

    st.subheader("Subscription Types")

    fig, ax = plt.subplots(figsize=(5, 3))

    df["subscription_type"].value_counts().plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("")
    ax.set_ylabel("Customers")

    plt.tight_layout()

    st.pyplot(fig)

st.divider()

# =====================================
# REGION + DEVICE
# =====================================

col1, col2 = st.columns(2)

with col1:

    st.subheader("Region Distribution")

    fig, ax = plt.subplots(figsize=(5, 3))

    df["region"].value_counts().plot(
        kind="barh",
        ax=ax
    )

    ax.set_xlabel("Customers")

    plt.tight_layout()

    st.pyplot(fig)

with col2:

    st.subheader("Device Usage")

    fig, ax = plt.subplots(figsize=(5, 3))

    df["device"].value_counts().plot(
        kind="barh",
        ax=ax
    )

    ax.set_xlabel("Customers")

    plt.tight_layout()

    st.pyplot(fig)

st.divider()

# =====================================
# HEATMAP
# =====================================

st.subheader("🔥 Correlation Heatmap")

corr_features = [
    "watch_hours",
    "avg_watch_time_per_day",
    "engagement_score",
    "last_login_days",
    "monthly_fee",
    "number_of_profiles"
]

corr = df[corr_features].corr()

fig, ax = plt.subplots(figsize=(5.5, 3.8))

heatmap = ax.imshow(
    corr,
    cmap="coolwarm"
)

ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))

ax.set_xticklabels(
    corr.columns,
    rotation=45,
    ha="right"
)

ax.set_yticklabels(corr.columns)

for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(
            j,
            i,
            f"{corr.iloc[i,j]:.2f}",
            ha="center",
            va="center",
            fontsize=8
        )

plt.colorbar(heatmap, shrink=0.8)

plt.tight_layout()

col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    st.pyplot(fig)

st.divider()

# =====================================
# CHURN BY SUBSCRIPTION
# =====================================

st.subheader("⚠️ Churn by Subscription Type")

churn_sub = pd.crosstab(
    df["subscription_type"],
    df["churned"]
)

fig, ax = plt.subplots(figsize=(5, 3))

churn_sub.plot(
    kind="bar",
    stacked=True,
    ax=ax,
    width=0.5
)

ax.set_ylabel("Customers")
ax.set_xlabel("")

plt.tight_layout()

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.pyplot(fig)

st.divider()

# =====================================
# INSIGHTS
# =====================================

st.success("""
### Key Insights

• Average Watch Time Per Day is the strongest predictor of churn.

• Engagement Score significantly influences customer retention.

• Customers inactive for long periods show higher churn risk.

• Watch Hours and Engagement Score are positively correlated.

• Lower activity levels generally increase churn probability.

• Subscription type impacts retention behavior.
""")