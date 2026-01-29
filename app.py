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

# Force Plotly to use a visible theme
pio.templates.default = "plotly_white"

# 1. Improved Data Fetching (Fixes the Multi-Index Graph Bug)
@st.cache_data(ttl=1800)
def fetch_stock_data(ticker, days_back=2000):
    end = dt.date.today()
    start = end - dt.timedelta(days=days_back)
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            return pd.DataFrame()
        
        # FIX: Flatten Multi-Index columns if they exist
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Ensure 'Close' is a Series and not a DataFrame
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
        return tf.keras.models.load_model('lstm_stock_model.h5')
    except Exception as e:
        st.error(f"Model load failed: {e}. Ensure 'lstm_stock_model.h5' is in your repo.")
        return None

# --- UI Setup ---
st.set_page_config(page_title="NSE Stock Predictor", layout="wide")
st.title("📈 Indian Stock Market Predictor (NSE)")
st.caption("LSTM Neural Network Demo • Skips Weekends for Predictions")

model = load_model()

if model is None:
    st.info("Please upload your 'lstm_stock_model.h5' file to the GitHub repository.")
    st.stop()

stocks_dict = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Infosys": "INFY.NS",
    "Nifty 50 Index": "^NSEI"
}

# Sidebar for controls
selected = st.sidebar.selectbox("Select Stock", list(stocks_dict.keys()), index=0)
ticker = stocks_dict[selected]
future_days = st.sidebar.slider("Days to Predict", 1, 30, 10)

if st.sidebar.button("Generate Prediction", type="primary"):
    with st.spinner(f"Processing {selected}..."):
        df = fetch_stock_data(ticker)
        
        if df.empty or len(df) < 100:
            st.error("Not enough data. Try a different ticker or check your connection.")
        else:
            # Display Metrics
            curr = df['Close'].iloc[-1]
            prev = df['Close'].iloc[-2]
            st.metric(f"Current {selected} Price", f"₹{curr:,.2f}", f"{curr-prev:,.2f}")

            # Scaling
            close_prices = df['Close'].values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled = scaler.fit_transform(close_prices)
            
            # --- PREDICTION LOGIC ---
            last_60 = scaled[-60:].reshape(1, 60, 1)
            preds_scaled = []
            temp_batch = last_60.copy()
            
            for _ in range(future_days):
                p = model.predict(temp_batch, verbose=0)
                preds_scaled.append(p[0, 0])
                # Roll the window
                temp_batch = np.append(temp_batch[:, 1:, :], [p.reshape(1, 1, 1)], axis=1)
            
            preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1))
            
            # REMEMBER WEEKEND DATA: Use Business Day range (bdate_range)
            # This ensures your prediction dates skip Saturdays and Sundays
            future_dates = pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1), periods=future_days)

            # --- VISUALIZATION ---
            col1, col2 = st.columns([2, 1])

            with col1:
                fig = go.Figure()
                # Historical
                fig.add_trace(go.Scatter(
                    x=df.index[-150:], 
                    y=df['Close'][-150:],
                    name='Historical', line=dict(color='#1f77b4')
                ))
                # Prediction
                fig.add_trace(go.Scatter(
                    x=future_dates, 
                    y=preds.flatten(),
                    name='Predicted', line=dict(color='orange', dash='dash', width=3)
                ))
                fig.update_layout(
                    title=f"{selected} Price Forecast",
                    xaxis_title="Date", yaxis_title="Price (₹)",
                    height=500, hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Forecast Table")
                pred_df = pd.DataFrame({
                    'Date': future_dates.strftime('%Y-%m-%d'),
                    'Price (₹)': preds.flatten().round(2)
                })
                st.dataframe(pred_df, use_container_width=True, hide_index=True)

else:
    st.info("Select a stock and click 'Generate Prediction' to begin.")
