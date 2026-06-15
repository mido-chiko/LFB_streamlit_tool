import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. Page Configuration & Logo
# -----------------------------------------------------------------------------
st.set_page_config(page_title="LFB Prediction Simulator", page_icon="🔮", layout="wide")
#st.logo("assets/lfb_logo.png")

st.title("Interactive Scenario Simulator")
st.markdown("Select a real historical incident, adjust key operational variables, and watch the LightGBM algorithm recalculate the expected response time in real-time.")
st.divider()

# -----------------------------------------------------------------------------
# 2. Data & Model Loading (Cached)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_simulator_resources():
    # Load the champion model
    model = joblib.load('models/lightgbm_regression_model.pkl')

    # Load the processed test set (269 columns)
    X_test_processed = pd.read_parquet('models/X_test_proc_deploy.parquet', engine='fastparquet')

    # Load the raw test set for human-readable context
    try:
        X_test_raw = pd.read_parquet('models/X_test_raw_deploy.parquet', engine='fastparquet')
    except:
        X_test_raw = X_test_processed.copy()

    # Load the target values
    try:
        y_test = pd.read_parquet('models/y_test_deploy.parquet', engine='fastparquet')
        if isinstance(y_test, pd.DataFrame):
            y_test = y_test.iloc[:, 0] # Extract series if saved as df
    except:
        # Fallback if y_test isn't easily loadable
        y_test = pd.Series([0]*len(X_test_processed), index=X_test_processed.index)

    return model, X_test_processed, X_test_raw, y_test

try:
    model, X_test_processed, X_test_raw, y_test = load_simulator_resources()
    data_loaded = True
except Exception as e:
    st.error(f"⚠️ Error loading resources: {e}")
    data_loaded = False

# -----------------------------------------------------------------------------
# 3. Interactive UI
# -----------------------------------------------------------------------------
if data_loaded:
    # --- Sidebar Controls ---
    st.sidebar.markdown("### 🎯 1. Select an Incident")

    # Let the user pick from the first 500 incidents in the test set
    max_idx = min(500, len(X_test_processed) - 1)
    incident_idx = st.sidebar.slider("Scroll through historical records:", min_value=0, max_value=max_idx, value=0)

    # Extract the specific row
    base_row_proc = X_test_processed.iloc[[incident_idx]].copy()
    base_row_raw = X_test_raw.iloc[incident_idx]
    actual_time = y_test.iloc[incident_idx]

    # Calculate baseline prediction
    baseline_pred = model.predict(base_row_proc)[0]

    # --- Top Row Context ---
    col1, col2, col3 = st.columns([1.5, 1, 1])

    with col1:
        st.markdown("#### 🚨 Incident Context")
        st.info(f"**Incident Type:** {base_row_raw.get('IncidentGroup', 'Unknown')}  \n"
                f"**Hour of Call:** {base_row_raw.get('HourOfCall', 'Unknown')}:00  \n"
                f"**Standardized Distance:** {base_row_proc['distance'].values[0]:.2f}")

    with col2:
        st.markdown("#### ⏱️ Actual Time")
        st.metric("Recorded Arrival", f"{int(actual_time)} sec")

    with col3:
        st.markdown("#### 🤖 Baseline Prediction")
        error = abs(baseline_pred - actual_time)
        error_color = "normal" if error <= 52.27 else "inverse" # 52.27 is your MAE
        st.metric(
            "Model Estimate",
            f"{int(baseline_pred)} sec",
            f"{int(baseline_pred - actual_time)} sec diff",
            delta_color=error_color
        )

    st.divider()

    # --- The "What-If" Engine ---
    st.markdown("### 🎛️ 2. What-If Scenario Tweak")
    st.markdown("Adjust the operational constraints below. The dial on the right will update in real-time, showing how logistical changes impact the LFB response targets.")

    control_col, chart_col = st.columns([1, 1.2])

    # Create an interactive copy of the row
    sim_row_proc = base_row_proc.copy()

    with control_col:
        st.markdown("<br>", unsafe_allow_html=True) # Spacing

        # Distance Tweak
        if 'distance' in sim_row_proc.columns:
            dist_multiplier = st.slider("🛣️ Adjust Distance to Incident (%)", min_value=10, max_value=300, value=100, step=5,
                                        help="Simulate the engine traveling a shorter or longer route.")
            sim_row_proc['distance'] = sim_row_proc['distance'] * (dist_multiplier / 100.0)

        # Easting Tweak
        if 'Easting_m' in sim_row_proc.columns:
            easting_shift = st.slider("🗺️ Shift Geographic Location (Easting)", min_value=-3.0, max_value=3.0, value=0.0, step=0.1,
                                      help="Simulate the incident occurring in a different geographic sector.")
            sim_row_proc['Easting_m'] = sim_row_proc['Easting_m'] + easting_shift

        # Hour of Call Tweak
        if 'HourOfCall' in sim_row_proc.columns:
            hour_shift = st.slider("🕒 Shift Time of Day", min_value=-3.0, max_value=3.0, value=0.0, step=0.1,
                                   help="Simulate the incident happening at a different traffic peak hour.")
            sim_row_proc['HourOfCall'] = sim_row_proc['HourOfCall'] + hour_shift

        # Calculate simulated prediction
        simulated_pred = model.predict(sim_row_proc)[0]

    with chart_col:
        # Build the Interactive Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=simulated_pred,
            title={'text': "Simulated Response Time", 'font': {'size': 20, 'color': '#1a3f6c'}},
            delta={'reference': baseline_pred, 'increasing': {'color': "#dc3545"}, 'decreasing': {'color': "#28a745"}},
            number={'suffix': " Sec", 'font': {'size': 40}},
            gauge={
                'axis': {'range': [None, max(800, simulated_pred + 100)], 'tickwidth': 1},
                'bar': {'color': "#1a3f6c"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 360], 'color': 'rgba(40, 167, 69, 0.2)'},    # Green: Under 6 mins
                    {'range': [360, 480], 'color': 'rgba(255, 193, 7, 0.2)'},  # Yellow: 6 to 8 mins
                    {'range': [480, 2000], 'color': 'rgba(220, 53, 69, 0.2)'}  # Red: Over 8 mins
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.75,
                    'value': actual_time # Shows where the ACTUAL incident time was on the dial
                }
            }
        ))

        fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # --- Bottom Comparative Analysis ---
    st.markdown("### 📊 Scenario Breakdown")

    # Dynamic text based on outcome
    delta = simulated_pred - baseline_pred
    if delta > 0:
        st.error(f"⚠️ **Logistical Delay:** These simulated changes increased the expected response time by **{int(delta)} seconds**.")
    elif delta < 0:
        st.success(f"✅ **Operational Optimization:** These simulated changes decreased the expected response time by **{int(abs(delta))} seconds**.")
    else:
        st.info("The model indicates the response time is resilient to these specific changes.")

    # Comparative Bar Chart
    fig_bar = go.Figure(data=[
        go.Bar(name='Actual Recorded Time', x=['Response Times'], y=[actual_time], marker_color='#6c757d'),
        go.Bar(name='Baseline ML Prediction', x=['Response Times'], y=[baseline_pred], marker_color='#1a3f6c'),
        go.Bar(name='Simulated ML Prediction', x=['Response Times'], y=[simulated_pred], marker_color='#ffc107' if delta > 0 else '#28a745')
    ])

    fig_bar.update_layout(
        barmode='group',
        height=300,
        yaxis_title="Seconds",
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_bar, use_container_width=True)
