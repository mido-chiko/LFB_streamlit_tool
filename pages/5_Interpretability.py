import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. Page Configuration & Logo
# -----------------------------------------------------------------------------
st.set_page_config(page_title="LFB Interpretability", layout="wide")
#st.logo("assets/lfb_logo.png")

st.title("Model Interpretability (SHAP)")
st.markdown("Understanding the inner workings of the Champion LightGBM Model. Shapley Additive exPlanations (SHAP) break down exactly how much each feature contributed to the final predicted attendance time.")
st.divider()

# -----------------------------------------------------------------------------
# 2. Data & SHAP Loading (Cached)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_shap_data():
    # Load the LightGBM Model and the pre-calculated SHAP values
    model = joblib.load('models/lightgbm_regression_model.pkl')
    shap_values = joblib.load('models/shap_values_deploy.pkl')

    # Attempt to load the engineered test set
    try:
        X_test_processed = pd.read_parquet('models/X_test_proc_deploy.parquet', engine='pyarrow')
    except FileNotFoundError:
        # If the processed file is missing, we raise an explicit error to prevent dimension crashes
        raise FileNotFoundError("Could not find 'data.parquet_X_test_processed'. Please save X_test_processed to parquet in your notebook.")

    # SHAP plotting is heavy; we use a representative sample for the dashboard
    sample_size = min(1000, len(X_test_processed))

    # Ensure our sample matches the length of the shap_values array
    if isinstance(shap_values, np.ndarray) and len(shap_values) > sample_size:
        indices = np.random.choice(len(X_test_processed), sample_size, replace=False)
        X_test_sample = X_test_processed.iloc[indices].copy()
        shap_sample = shap_values[indices]
    else:
        X_test_sample = X_test_processed.copy()
        shap_sample = shap_values

    # Calculate expected value (Base Value) for Waterfall plots
    explainer = shap.TreeExplainer(model)
    base_value = explainer.expected_value
    if isinstance(base_value, np.ndarray):
        base_value = base_value[0]

    return X_test_sample, shap_sample, base_value

# Handle Loading States
try:
    X_test_sample, shap_sample, expected_value = load_shap_data()

    # STRICT DIMENSION CHECK
    if X_test_sample.shape[1] != shap_sample.shape[1]:
        st.error(f"❌ **Dimension Mismatch:** Your SHAP array has {shap_sample.shape[1]} columns, but your test dataset has {X_test_sample.shape[1]} columns. Please ensure you uploaded the fully encoded `X_test_processed`.")
        data_loaded = False
    else:
        data_loaded = True

except FileNotFoundError as e:
    st.error(f"❌ **Missing File:** {e}")
    data_loaded = False
except Exception as e:
    st.error(f"❌ **Error Loading Data:** {e}")
    data_loaded = False

# -----------------------------------------------------------------------------
# 3. Interactive Visualization Tabs
# -----------------------------------------------------------------------------
if data_loaded:
    tab1, tab2, tab3 = st.tabs(["Global Summary", "Feature Dependence", "Local Incident Breakdown"])

    with tab1:
        st.markdown("### Global Feature Impact (Summary Plot)")
        st.markdown("Features at the top have the largest impact. The color represents the actual value of the feature (Red = High, Blue = Low), and the horizontal location shows whether that value increased or decreased the predicted response time.")

        c1, c2, c3 = st.columns([1, 10, 1])
        with c2:
            fig_summary, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_sample, X_test_sample, feature_names=X_test_sample.columns, show=False)
            st.pyplot(fig_summary, bbox_inches='tight')
            plt.clf()

    with tab2:
        st.markdown("### Feature Dependence Analysis")
        st.markdown("Observe how specific feature values non-linearly affect the response time. For example, see how `distance` scales the predicted seconds, and how it interacts with other operational variables.")

        # Define the top features to select from based on your LightGBM feature importance
        top_features = ["distance", "USRN", "HourOfCall", "Easting_m", "Northing_m", "CalYear", "DelayCodeId"]
        available_features = [f for f in top_features if f in X_test_sample.columns]

        if available_features:
            selected_feature = st.selectbox("Select a feature to analyze:", available_features)

            c1, c2, c3 = st.columns([1, 10, 1])
            with c2:
                fig_dep, ax = plt.subplots(figsize=(10, 6))
                shap.dependence_plot(selected_feature, shap_sample, X_test_sample, interaction_index="auto", show=False, ax=ax)
                st.pyplot(fig_dep, bbox_inches='tight')
                plt.clf()
        else:
            st.warning("Top features not found in the dataset column names.")

    with tab3:
        st.markdown("### Local Explanation (Waterfall Plot)")
        st.markdown("Drill down into a specific, individual emergency incident to see exactly how the LightGBM algorithm arrived at its final prediction starting from the baseline average.")

        # Slider to pick a specific incident from our sample
        incident_idx = st.slider("Select Incident Index", min_value=0, max_value=len(X_test_sample)-1, value=0)

        c1, c2, c3 = st.columns([1, 10, 1])
        with c2:
            # Create a localized Explanation object for the specific index
            local_exp = shap.Explanation(
                values=shap_sample[incident_idx],
                base_values=expected_value,
                data=X_test_sample.iloc[incident_idx, :],
                feature_names=X_test_sample.columns
            )

            fig_waterfall, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(local_exp, show=False)
            st.pyplot(fig_waterfall, bbox_inches='tight')
            plt.clf()
