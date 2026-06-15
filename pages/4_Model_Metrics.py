import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. Page Configuration & Logo
# -----------------------------------------------------------------------------
st.set_page_config(page_title="LFB Model Metrics", page_icon="📊", layout="wide")
#st.logo("assets/lfb_logo.png")

st.title("Machine Learning Evaluation")
st.markdown("Comparative performance metrics across regularized linear models, tree-based ensembles, and deep learning architectures.")
st.divider()

# -----------------------------------------------------------------------------
# 2. Static Metrics Data Definition
# -----------------------------------------------------------------------------
@st.cache_data
def load_model_metrics():
    # Pre-calculated metrics ensure the dashboard remains lightning-fast.
    data = {
        "Phase": [
            "1: Baseline", "1: Baseline", "1: Baseline", "1: Baseline",
            "2: Ensembles", "2: Ensembles",
            "3: Deep Learning"
        ],
        "Algorithm": [
            "Linear Regression", "Ridge (L2)", "Lasso (L1)", "ElasticNet",
            "XGBoost", "LightGBM",
            "Deep Learning (Keras)"
        ],
        "Test RMSE": [91.71, 91.50, 95.20, 96.10, 73.65, 72.36, 74.12],
        "Test MAE": [63.19, 63.10, 65.20, 66.10, 49.80, 48.80, 49.50],
        "Test R²": [0.547, 0.548, 0.510, 0.490, 0.695, 0.706, 0.691],
        "Type": [
            "Linear", "Linear", "Linear", "Linear",
            "Tree", "Tree",
            "Neural Network"
        ]
    }
    return pd.DataFrame(data)

metrics_df = load_model_metrics()

# -----------------------------------------------------------------------------
# 3. High-Level Winner Highlight
# -----------------------------------------------------------------------------
st.markdown("### 🏆 The Champion Model: LightGBM")
st.markdown("Following the removal of noisy post-incident features (e.g., `PumpMinutesRounded`), the LightGBM framework achieved the highest predictive accuracy. It successfully mapped over 70% of the variance in emergency response times without succumbing to severe overfitting.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Test RMSE", "72.36 s", "- 19.35s from baseline", delta_color="inverse")
col2.metric("Test MAE", "48.80 s", "- 14.39s from baseline", delta_color="inverse")
col3.metric("Test R²", "0.706", "+ 0.159 from baseline")
col4.metric("Train/Test R² Delta", "0.055", "Excellent Generalization", delta_color="normal")

st.divider()

# -----------------------------------------------------------------------------
# 4. Interactive Evaluation Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Performance Leaderboard", "📈 Visual Comparison", "🧠 DL vs Ensembles"])

with tab1:
    st.markdown("### Model Evaluation Matrix")
    st.markdown("A comprehensive breakdown of all algorithms tested throughout the project phases.")

    # Format the dataframe for display
    display_df = metrics_df.copy()
    display_df['Test RMSE'] = display_df['Test RMSE'].apply(lambda x: f"{x:.2f} s")
    display_df['Test MAE'] = display_df['Test MAE'].apply(lambda x: f"{x:.2f} s")
    display_df['Test R²'] = display_df['Test R²'].apply(lambda x: f"{x:.3f}")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

with tab2:
    st.markdown("### Metrics Visualization")

    c1, c2 = st.columns(2)
    with c1:
        # RMSE Chart (Lower is better)
        fig_rmse = px.bar(
            metrics_df.sort_values('Test RMSE', ascending=False),
            x='Test RMSE',
            y='Algorithm',
            color='Type',
            orientation='h',
            title='Root Mean Squared Error (Lower is Better)',
            color_discrete_map={'Linear': '#6c757d', 'Tree': '#28a745', 'Neural Network': '#17a2b8'}
        )
        fig_rmse.add_vline(x=72.36, line_dash="dash", line_color="red", annotation_text="Best: 72.36s")
        st.plotly_chart(fig_rmse, use_container_width=True)

    with c2:
        # R-Squared Chart (Higher is better)
        fig_r2 = px.bar(
            metrics_df.sort_values('Test R²', ascending=True),
            x='Test R²',
            y='Algorithm',
            color='Type',
            orientation='h',
            title='Variance Explained (R² - Higher is Better)',
            color_discrete_map={'Linear': '#6c757d', 'Tree': '#28a745', 'Neural Network': '#17a2b8'}
        )
        fig_r2.add_vline(x=0.706, line_dash="dash", line_color="red", annotation_text="Best: 0.706")
        st.plotly_chart(fig_r2, use_container_width=True)

with tab3:
    st.markdown("### 🧠 Deep Learning vs. Tree-Based Ensembles")

    col_text, col_chart = st.columns([1, 1])

    with col_text:
        st.markdown("""
        **Architectural Observations:**
        * **Tabular Data Dominance:** While the Keras Deep Learning model performed exceptionally well, **LightGBM** retained a slight edge. This aligns with industry consensus that gradient-boosted decision trees often outperform standard neural networks on purely tabular, high-cardinality data.
        * **Training Efficiency:** LightGBM achieved its optimal state at iteration 491, utilizing leaf-wise tree growth. This was computationally lighter to train and tune than the neural network's backpropagation across 100 epochs.
        * **Interpretability:** Tree-based models allow for precise SHAP value extraction (explored in the next tab), making the LightGBM model much more actionable for London Fire Brigade stakeholders than the "black box" nature of deep learning.
        """)

    with col_chart:
        # Radar chart comparing the top 3 models
        top_models = metrics_df[metrics_df['Algorithm'].isin(['XGBoost', 'LightGBM', 'Deep Learning (Keras)'])]

        fig_radar = go.Figure()

        for index, row in top_models.iterrows():
            # Normalize metrics for radar chart visual parity (R2 is 0-1, Error needs inversion/scaling)
            normalized_r2 = row['Test R²'] / 0.706
            normalized_mae = 48.80 / row['Test MAE'] # Invert so larger area = better
            normalized_rmse = 72.36 / row['Test RMSE'] # Invert so larger area = better

            fig_radar.add_trace(go.Scatterpolar(
                r=[normalized_r2, normalized_mae, normalized_rmse, normalized_r2],
                theta=['R² Score', 'MAE (Inverted)', 'RMSE (Inverted)', 'R² Score'],
                fill='toself',
                name=row['Algorithm']
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=False, range=[0.85, 1.05])),
            showlegend=True,
            title="Top 3 Models Relative Footprint (Larger Area = Better)",
            margin=dict(t=50, b=0, l=0, r=0)
        )
        st.plotly_chart(fig_radar, use_container_width=True)
