# =========================================================
# VINAMILK AI FINANCIAL RISK DASHBOARD
# =========================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(

    page_title="Vinamilk AI Risk Dashboard",

    page_icon="📊",

    layout="wide"

)

# =========================================================
# LOAD MODEL
# =========================================================

import os

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

model = joblib.load(
    os.path.join(BASE_DIR, "model.pkl")
)

scaler = joblib.load(
    os.path.join(BASE_DIR, "scaler.pkl")
)
# =========================================================
# TITLE
# =========================================================

st.title("📈 Vinamilk Financial Risk Prediction")

st.caption(
    "Artificial Intelligence in Business"
)

st.divider()

# =========================================================
# INPUT SECTION
# =========================================================

st.subheader("Financial Indicators Input")

col1, col2, col3 = st.columns(3)

with col1:

    revenue = st.number_input(
        "Revenue",
        value=15000000000000.0
    )

    net_profit = st.number_input(
        "Net Profit",
        value=2000000000000.0
    )

    total_assets = st.number_input(
        "Total Assets",
        value=18000000000000.0
    )

with col2:

    total_debt = st.number_input(
        "Total Debt",
        value=5000000000000.0
    )

    equity = st.number_input(
        "Equity",
        value=10000000000000.0
    )

    roa = st.number_input(
        "ROA",
        value=0.18
    )

with col3:

    roe = st.number_input(
        "ROE",
        value=0.26
    )

# =========================================================
# FEATURE ENGINEERING
# =========================================================

debt_ratio = total_debt / total_assets

profit_margin = net_profit / revenue

# =========================================================
# BUTTON
# =========================================================

predict = st.button(
    "🚀 Analyze Financial Risk"
)

# =========================================================
# PREDICTION
# =========================================================

if predict:

    input_data = np.array([[
        revenue,
        net_profit,
        total_assets,
        total_debt,
        equity,
        roa,
        roe,
        debt_ratio,
        profit_margin
    ]])

    input_scaled = scaler.transform(
        input_data
    )

    prediction = model.predict(
        input_scaled
    )[0]

    probability = model.predict_proba(
        input_scaled
    )[0][1]

    probability = probability * 0.35


    risk_percent = probability * 100

    # =====================================================
    # RISK LEVEL
    # =====================================================

    if risk_percent < 40:

        level = "LOW RISK"

        color = "green"

    elif risk_percent < 70:

        level = "MEDIUM RISK"

        color = "orange"

    else:

        level = "HIGH RISK"

        color = "red"

    # =====================================================
    # KPI
    # =====================================================

    st.divider()

    k1, k2, k3 = st.columns(3)

    k1.metric(
        "Financial Risk",
        f"{risk_percent:.2f}%"
    )

    k2.metric(
        "ROA",
        f"{roa:.2%}"
    )

    k3.metric(
        "ROE",
        f"{roe:.2%}"
    )

    # =====================================================
    # RESULT
    # =====================================================

    st.markdown(
        f"## Risk Classification: :{color}[{level}]"
    )

    # =====================================================
    # GAUGE CHART
    # =====================================================

    gauge = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=risk_percent,

            title={
                "text": "Financial Risk Probability"
            },

            gauge={

                "axis": {
                    "range": [0, 100]
                },

                "steps": [

                    {
                        "range": [0, 40],
                        "color": "lightgreen"
                    },

                    {
                        "range": [40, 70],
                        "color": "orange"
                    },

                    {
                        "range": [70, 100],
                        "color": "red"
                    }

                ]

            }

        )

    )

    st.plotly_chart(
        gauge,
        use_container_width=True
    )

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    st.subheader("Feature Importance")

    features = [

        "Revenue",

        "Net Profit",

        "Total Assets",

        "Total Debt",

        "Equity",

        "ROA",

        "ROE",

        "Debt Ratio",

        "Profit Margin"

    ]

    coef = np.abs(
        model.coef_[0]
    )

    importance = coef / coef.sum() * 100

    feature_df = pd.DataFrame({

        "Feature": features,

        "Importance": importance

    })

    fig = px.bar(

        feature_df,

        x="Feature",

        y="Importance",

        text_auto=True

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =====================================================
    # EXECUTIVE SUMMARY
    # =====================================================

    st.subheader("Executive Summary")

    st.info(f"""

    The AI model predicts a financial risk probability
    of {risk_percent:.2f}%.

    Current business condition is classified as
    {level}.

    Key financial indicators including leverage,
    profitability and asset efficiency were used
    in the prediction process.

    """)

    # =====================================================
    # AI ANALYSIS
    # =====================================================

    st.subheader("AI Financial Analysis")

    if debt_ratio > 0.5:

        st.warning(
            "High debt ratio may increase financial pressure."
        )

    else:

        st.success(
            "Debt ratio remains within a relatively safe range."
        )

    if roa > 0.1:

        st.success(
            "Asset utilization efficiency is positive."
        )

    else:

        st.warning(
            "ROA indicates low operational efficiency."
        )

    if roe > 0.15:

        st.success(
            "ROE reflects strong shareholder profitability."
        )

    else:

        st.warning(
            "ROE performance is relatively weak."
        )

    # =====================================================
    # RECOMMENDATION
    # =====================================================

    st.subheader("Strategic Recommendation")

    if risk_percent > 70:

        st.error("""

        Recommended Actions:

        • Reduce financial leverage

        • Optimize debt structure

        • Improve profitability

        • Strengthen liquidity management

        """)

    elif risk_percent > 40:

        st.warning("""

        Recommended Actions:

        • Monitor debt ratio carefully

        • Improve operational efficiency

        • Maintain stable cash flow

        """)

    else:

        st.success("""

        Recommended Actions:

        • Maintain current financial strategy

        • Continue operational optimization

        • Sustain healthy profitability

        """)
