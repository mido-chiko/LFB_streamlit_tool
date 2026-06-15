import streamlit as st

# -----------------------------------------------------------------------------
# 1. Page Configuration & Logo
# -----------------------------------------------------------------------------
st.set_page_config(page_title="LFB Project Home", page_icon="🚒", layout="wide")
st.logo("assets/lfb_logo.png")
st.image("assets/lfb_logo.png", width=300)


# -----------------------------------------------------------------------------
# 2. Header & Title
# -----------------------------------------------------------------------------
st.title("London Fire Brigade: Predictive Analytics & Modeling")
st.markdown("### Capstone Project: Estimating Average Attendance Time")
st.markdown("*Data Scientist Training Program — Liora*")
st.divider()

# -----------------------------------------------------------------------------
# 3. Executive Summary
# -----------------------------------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🎯 Project Objective")
    st.markdown("""
    This project details a comprehensive, end-to-end machine learning pipeline developed to accurately predict the `Average_AttendanceTimeSeconds` for London Fire Brigade (LFB) incidents. 
    
    By transforming raw operational logs into a robust predictive model, this tool provides strategic value for concurrent resource planning and emergency dispatch optimization.
    """)
    
    st.markdown("### 🏗️ Methodology & Pipeline")
    st.markdown("""
    The project adheres to a strict, milestone-driven methodology:
    * **Phase 1: Scoping & Data Exploration** — Processing >1.9 million valid records from combined Incident and Mobilisation datasets.
    * **Phase 2: Feature Engineering** — Spatial routing (Distance calculations), logistical enrichment (Delay Codes), and rigorous target leakage prevention.
    * **Phase 3: Iterative Modeling** — Establishing regularized linear baselines before advancing to complex Tree-Based Ensembles and Deep Learning architectures.
    * **Phase 4: Time Series Forecasting** — Aggregating data for daily and monthly temporal prediction.
    """)

with col2:
    st.info("#### 📊 Dataset Scale")
    st.metric(label="Total Processed Records", value="> 1.9 Million")
    st.metric(label="Raw Feature Space", value="534 Features")
    st.metric(label="Optimized Feature Matrix", value="269 Features")

st.divider()

# -----------------------------------------------------------------------------
# 4. Key Achievements / Final Results
# -----------------------------------------------------------------------------
st.markdown("### 🏆 Apex Modeling Results")
st.markdown("The deployment of the **LightGBM** framework represents the apex of the predictive pipeline. After removing post-incident logistical noise, the model efficiently mapped spatial-temporal interactions via leaf-wise tree growth, achieving exceptional stability between training and testing data.")

r1, r2, r3 = st.columns(3)

with r1:
    st.success("#### High Predictive Power")
    st.markdown("Over **70%** of the complex variance in London's emergency response times successfully mapped.")
    st.metric(label="Test R² Score", value="0.706")

with r2:
    st.success("#### Operational Accuracy")
    st.markdown("Predicting the arrival of fire brigade resources to within less than 49 seconds.")
    st.metric(label="Final MAE", value="48.80 Sec")

with r3:
    st.success("#### Baseline Improvement")
    st.markdown("Massive reduction in error compared to the initial un-engineered regularized linear baselines.")
    st.metric(label="RMSE Reduction", value="- 59.35 Sec", delta="Improved from 131.71s to 72.36s", delta_color="inverse")

# -----------------------------------------------------------------------------
# 5. Navigation Guide
# -----------------------------------------------------------------------------
st.divider()
st.markdown("### 🧭 Application Navigation")
st.markdown("""
Use the sidebar to explore the complete lifecycle of this project:
1. **Data Dictionary:** Definitions of the LFB operational features.
2. **Dashboard:** Interactive EDA, geospatial mapping, and historical incident tracking.
3. **Time Series:** Daily and monthly emergency forecasting.
4. **Model Metrics:** Technical evaluation of LR, ElasticNet, Lasso, XGBoost, LightGBM, and Deep Learning.
5. **Interpretability:** SHAP value analysis detailing the impact of top features like Distance, USRN, and HourOfCall.
6. **Prediction Simulator:** Live, interactive inference using the optimized model.
""")