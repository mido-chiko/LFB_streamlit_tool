import streamlit as st
import pandas as pd
import numpy as np
import datetime
import joblib
import plotly.graph_objects as go
from sklearn.metrics import root_mean_squared_error
import os
# -----------------------------------------------------------------------------
# 1. Page Configuration & Logo
# -----------------------------------------------------------------------------
st.set_page_config(page_title="LFB Time Series", layout="wide")
st.title("Time Series Analysis & Forecasting")
st.markdown("Predictive forecasting of London Fire Brigade average attendance times using optimized Prophet models for daily and monthly aggregations.")
st.divider()
# -----------------------------------------------------------------------------
# 2. Data & Model Loading (Cached)
# -----------------------------------------------------------------------------
@st.cache_data
def load_ts_data():
    # Intelligently find the files whether they are in the root or the models folder
    daily_path = 'models/data.parquet_daily_series_df' if os.path.exists('models/data.parquet_daily_series_df') else 'data.parquet_daily_series_df'
    monthly_path = 'models/data.parquet_monthly_new' if os.path.exists('models/data.parquet_monthly_new') else 'data.parquet_monthly_new'

    # Load Daily Data
    daily_df = pd.read_parquet(daily_path, engine='fastparquet')
    df_daily = daily_df.set_index('DateOfCall').reset_index()
    df_daily = df_daily.rename(columns={"DateOfCall": "ds", "Average_AttendanceTimeSeconds": "y"})

    # Load Monthly Data
    monthly_df = pd.read_parquet(monthly_path, engine='fastparquet')
    df_monthly = monthly_df.rename(columns={"DateTime": "ds", "Average_AttendanceTimeSeconds": "y"})

    return df_daily, df_monthly

@st.cache_resource
def load_ts_models():
    # Intelligently find the models whether they are in the root or the models folder
    daily_model_path = 'models/TS_daily_ph_model.pkl' if os.path.exists('models/TS_daily_ph_model.pkl') else 'TS_daily_ph_model.pkl'
    monthly_model_path = 'models/monthly_TS_model.pkl' if os.path.exists('models/monthly_TS_model.pkl') else 'monthly_TS_model.pkl'

    # Load Pre-trained Prophet Models
    daily_model = joblib.load(daily_model_path)
    monthly_model = joblib.load(monthly_model_path)
    return daily_model, monthly_model

@st.cache_data
def generate_forecasts(_model, periods, freq):
    # Generate future dataframe and predict
    future = _model.make_future_dataframe(periods=periods, freq=freq)
    predictions = _model.predict(future)
    return predictions

# --- Execute Loading and Calculations ---
df_daily, df_monthly = load_ts_data()
model_daily, model_monthly = load_ts_models()

# Generate Predictions
pred_daily = generate_forecasts(model_daily, periods=727, freq='D')
pred_monthly = generate_forecasts(model_monthly, periods=24, freq='ME')

# Define test sets and calculate RMSE
test_daily = df_daily.iloc[-727:]
rmse_daily = root_mean_squared_error(test_daily['y'], pred_daily['yhat'].tail(727))

test_monthly = df_monthly.iloc[-24:]
rmse_monthly = root_mean_squared_error(test_monthly['y'], pred_monthly['yhat'].tail(24))

@st.cache_resource
def load_ts_models():
    # Intelligently find the models whether they are in the root or the models folder
    daily_model_path = 'models/TS_daily_ph_model.pkl' if os.path.exists('models/TS_daily_ph_model.pkl') else 'TS_daily_ph_model.pkl'
    monthly_model_path = 'models/monthly_TS_model.pkl' if os.path.exists('models/monthly_TS_model.pkl') else 'monthly_TS_model.pkl'

    # Load Pre-trained Prophet Models
    daily_model = joblib.load(daily_model_path)
    monthly_model = joblib.load(monthly_model_path)
    return daily_model, monthly_model

# -----------------------------------------------------------------------------
# 3. Reusable Plotting Function
# -----------------------------------------------------------------------------
def plot_interactive_forecast(df_actual, df_pred, test_size, title, y_label, cutoff_date=None):
    fig = go.Figure()

    # Actual Values (Full timeline)
    fig.add_trace(go.Scatter(
        x=df_actual['ds'], y=df_actual['y'],
        mode='lines', name='Actual Values',
        line=dict(color='#28a745', width=2)
    ))

    # Predicted Values (Test period)
    test_dates = df_actual['ds'].iloc[-test_size:]
    test_preds = df_pred['yhat'].tail(test_size)
    lower_bound = df_pred['yhat_lower'].tail(test_size)
    upper_bound = df_pred['yhat_upper'].tail(test_size)

    fig.add_trace(go.Scatter(
        x=test_dates, y=test_preds,
        mode='lines', name='Predictions',
        line=dict(color='#ffc107', width=2, dash='dash')
    ))

    # Confidence Interval
    fig.add_trace(go.Scatter(
        x=pd.concat([test_dates, test_dates[::-1]]),
        y=pd.concat([upper_bound, lower_bound[::-1]]),
        fill='toself',
        fillcolor='rgba(255, 193, 7, 0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name='80% Confidence Interval'
    ))

    # Train/Test Split Line
    split_date = df_actual['ds'].iloc[-test_size]
    # FIX: Convert datetime to ms timestamp float to bypass Pandas math errors
    split_date_ms = pd.to_datetime(split_date).timestamp() * 1000

    fig.add_vline(x=split_date_ms, line_width=2, line_dash="dash", line_color="red",
                  annotation_text="Train/Test Split", annotation_position="top left")

    # Custom Cutoff Line
    if cutoff_date:
        # Convert custom date to ms timestamp float as well
        cutoff_date_ms = pd.to_datetime(cutoff_date).timestamp() * 1000
        fig.add_vline(x=cutoff_date_ms, line_width=2, line_color="red",
                      annotation_text="Key Event", annotation_position="bottom right")

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title=y_label,
        hovermode='x unified',
        margin=dict(t=50, b=0, l=0, r=0)
    )
    return fig
# -----------------------------------------------------------------------------
# 4. Forecasting Dashboards (Tabs)
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Daily Forecast", "Monthly Forecast", "Trend & Seasonality"])

with tab1:
    st.markdown("### Daily Average Attendance Time Forecast")

    col1, col2 = st.columns([1, 3])
    with col1:
        st.info("#### Model Evaluation")
        st.metric(label="Testing RMSE", value=f"{rmse_daily:.2f} Sec")
        st.markdown("**Test Horizon:** 727 Days")
        st.markdown("**Algorithm:** Prophet")

    with col2:
        # Plot using our interactive function
        fig_d = plot_interactive_forecast(
            df_actual=df_daily,
            df_pred=pred_daily,
            test_size=727,
            title='Optimized Prophet Model: Daily Predictions',
            y_label='Avg Attendance Time (Sec)',
            cutoff_date=datetime.date(2023, 11, 4)
        )
        st.plotly_chart(fig_d, use_container_width=True)

with tab2:
    st.markdown("### Monthly Average Attendance Time Forecast")

    col1, col2 = st.columns([1, 3])
    with col1:
        st.info("#### Model Evaluation")
        st.metric(label="Testing RMSE", value=f"{rmse_monthly:.2f} Sec")
        st.markdown("**Test Horizon:** 24 Months")
        st.markdown("**Algorithm:** Prophet")

    with col2:
        fig_m = plot_interactive_forecast(
            df_actual=df_monthly,
            df_pred=pred_monthly,
            test_size=24,
            title='Optimized Prophet Model: Monthly Predictions',
            y_label='Avg Attendance Time (Sec)'
        )
        st.plotly_chart(fig_m, use_container_width=True)

with tab3:
    st.markdown("Time Series Decomposition (Daily Model)")
    st.markdown("Breaking down the historical data into structural components: overall trend, weekly variations, and yearly seasonality.")

    try:
        from prophet.plot import plot_components_plotly
        # Prophet has a native Plotly component generator
        fig_comp = plot_components_plotly(model_daily, pred_daily)

        # Prophet returns a list of figures or a subplot figure depending on version.
        # Ensure it renders nicely in Streamlit
        fig_comp.update_layout(height=800, margin=dict(t=30, b=30, l=0, r=0))
        st.plotly_chart(fig_comp, use_container_width=True)
    except ImportError:
        st.warning("To view interactive components, ensure `prophet` is installed in your Streamlit environment.")
