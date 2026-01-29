import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import datetime as dt
import time

# Force Plotly to use a visible theme for Streamlit Cloud
pio.templates.default = "plotly_white"

# 1. Improved Data Fetching (Fixes Multi-Index Graph Bug)
@st.cache_data(ttl=1800)
def fetch_stock_data(ticker, days_back=2000):
    end = dt.date.today()
    start = end - dt.timedelta(days=days_back)
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            return pd.DataFrame()
        
        # FIX: Flatten Multi-Index columns if they exist (yfinance 0.2.x+ behavior)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Ensure 'Close' is treated as a single column Series
        if isinstance(df['Close'], pd.DataFrame):
            df['Close'] = df['Close'].iloc[:, 0]
            
        return df
    except Exception as e:
        st.error(f"yfinance error: {e}")
        return pd.DataFrame()

# 2. Model Loading
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        # File must be in the root of your GitHub repo
        return tf.keras.models.load_model('lstm_stock_model.h5')
    except Exception as e:
        st.error(f"Model load failed: {e}. Check if 'lstm_stock_model.h5' is uploaded.")
        return None

# --- UI Setup ---
st.set_page_config(page_title="NSE Stock Predictor", layout="wide")
st.title("📈 Indian Stock Market Predictor (NSE)")
st.caption("LSTM Neural Network • Handles Weekends • Interactive Plotly Charts")

model = load_model()

if model is None:
    st.warning("Application stopped: Model file missing.")
    st.stop()

stocks_dict = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Infosys": "INFY.NS",
    "Nifty 50 Index": "^NSEI"
}

# Sidebar for user controls
st.sidebar.header("Settings")
selected = st.sidebar.selectbox("Select Stock", list(stocks_dict.keys()), index=0)
ticker = stocks_dict[selected]
future_days = st.sidebar.slider("Days to Predict", 1, 30, 10)

if st.sidebar.button("Generate Prediction", type="primary"):
    with st.spinner(f"Analyzing {selected}..."):
        df = fetch_stock_data(ticker)
        
        if df.empty or len(df) < 100:
            st.error("Insufficient data found for this ticker.")
        else:
            # Current Price Metrics
            curr = df['Close'].iloc[-1]
            prev = df['Close'].iloc[-2]
            st.metric(f"Current {selected} Price", f"₹{curr:,.2f}", f"{curr-prev:,.2f}")

            # Prepare Scaling
            close_prices = df['Close'].values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled = scaler.fit_transform(close_prices)
            
            # --- FIXED PREDICTION LOGIC (Fixes ValueError) ---
            last_60 = scaled[-60:].reshape(1, 60, 1)
            preds_scaled = []
            temp_batch = last_60.copy()
            
            for _ in range(future_days):
                # Predict next value
                p = model.predict(temp_batch, verbose=0)
                preds_scaled.append(p[0, 0])
                
                # Reshape prediction to match batch dimensions (1, 1, 1)
                p_reshaped = p.reshape(1, 1, 1)
                
                # Slide window: Remove oldest, add newest prediction
                temp_batch = np.concatenate((temp_batch[:, 1:, :], p_reshaped), axis=1)
            
            # Transform back to actual price
            preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1))
            
            # WEEKEND LOGIC: Use Business Day range to skip Saturdays/Sundays
            future_dates = pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1), periods=future_days)

            # --- VISUALIZATION ---
            col1, col2 = st.columns([2, 1])

            with col1:
                fig = go.Figure()
                # Historical Price Trace
                fig.add_trace(go.Scatter(
                    x=df.index[-150:], 
                    y=df['Close'][-150:],
                    name='Historical', 
                    line=dict(color='#1f77b4', width=2)
                ))
                # Prediction Trace
                fig.add_trace(go.Scatter(
                    x=future_dates, 
                    y=preds.flatten(),
                    name='Predicted', 
                    line=dict(color='#ff7f0e', dash='dash', width=3)
                ))
                fig.update_layout(
                    title=f"{selected} Price Forecast",
                    xaxis_title="Date", 
                    yaxis_title="Price (₹)",
                    height=550, 
                    hovermode="x unified",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Price Table")
                pred_df = pd.DataFrame({
                    'Date': future_dates.strftime('%d %b, %Y'),
                    'Price (₹)': preds.flatten().round(2)
                })
                st.dataframe(pred_df, use_container_width=True, hide_index=True)

else:
    st.info("Select a stock from the sidebar and click 'Generate Prediction'.")
