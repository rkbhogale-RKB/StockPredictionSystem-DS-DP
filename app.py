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

# Force Plotly to use a safe, visible theme (fixes black/blank charts on Streamlit Cloud)
pio.templates.default = "plotly_white"

# Cache data fetch to avoid rate limits
@st.cache_data(ttl=1800)  # cache for 30 minutes
def fetch_stock_data(ticker, days_back=2000, retries=3):
    end = dt.date.today()
    start = end - dt.timedelta(days=days_back)
    for attempt in range(retries):
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if not df.empty:
                return df
        except Exception as e:
            if "Rate" in str(e) or "Many Requests" in str(e):
                wait_time = 5 * (attempt + 1)
                st.warning(f"yfinance rate limit — retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise e
    st.error("Could not fetch data — rate limit or other issue. Try again later.")
    return pd.DataFrame()

# Load pre-trained model (upload 'lstm_stock_model.h5' to your repo!)
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        return tf.keras.models.load_model('lstm_stock_model.h5')
    except Exception as e:
        st.error(f"Model load failed: {e}")
        return None

model = load_model()

if model is None:
    st.stop()  # stop if model missing

st.set_page_config(page_title="NSE Stock Predictor", layout="wide")
st.title("Indian Stock Market Price Prediction (NSE) - LSTM Demo")
st.caption("Educational project only • Not financial advice • Predictions are illustrative")
st.markdown("**Disclaimer**: This is for learning purposes. Stock prices are influenced by many unpredictable factors.")

stocks_dict = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Infosys": "INFY.NS",
    "Nifty 50 Index": "^NSEI"
}

selected = st.selectbox("Select Stock", list(stocks_dict.keys()), index=0)
ticker = stocks_dict[selected]

future_days = st.slider("Predict next how many days?", min_value=1, max_value=30, value=10)

if st.button("Generate Prediction", type="primary"):
    with st.spinner(f"Fetching {selected} data & making predictions..."):
        df = fetch_stock_data(ticker)
        
        if df.empty or len(df) < 100:
            st.error("Not enough data available. Try another stock or wait for rate limit cooldown.")
        else:
            close_prices = df['Close'].values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled = scaler.fit_transform(close_prices)
            
            # Historical chart (last ~2 years)
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(
                x=df.index[-400:],
                y=df['Close'][-400:],
                mode='lines',
                name='Historical Close',
                line=dict(color='blue', width=2)
            ))
            fig_hist.update_layout(
                title=f"{selected} - Historical Closing Price",
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                height=500,
                margin=dict(l=40, r=40, t=60, b=40)
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Predict future prices
            last_60 = scaler.transform(df['Close'].values[-60:].reshape(-1, 1))
            batch = last_60.reshape(1, 60, 1)
            preds_scaled = []
            
            for _ in range(future_days):
                p = model.predict(batch, verbose=0)
                preds_scaled.append(p[0, 0])
                batch = np.roll(batch, -1, axis=1)
                batch[0, -1, 0] = p[0, 0]
            
            preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1))
            
            future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=future_days)
            
            # Prediction table
            pred_df = pd.DataFrame({
                'Date': future_dates.strftime('%Y-%m-%d'),
                'Predicted Close (₹)': preds.flatten().round(2)
            })
            st.subheader(f"Predicted Prices for Next {future_days} Days")
            st.dataframe(pred_df.style.format({"Predicted Close (₹)": "₹{:,.2f}"}), use_container_width=True)
            
            # Forecast chart
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=df.index[-200:],
                y=df['Close'][-200:],
                mode='lines',
                name='Historical',
                line=dict(color='blue', width=2)
            ))
            fig_pred.add_trace(go.Scatter(
                x=future_dates,
                y=preds.flatten(),
                mode='lines+markers',
                name='Predicted',
                line=dict(color='orange', width=2, dash='dash')
            ))
            fig_pred.update_layout(
                title=f"{selected} - Historical + {future_days}-Day Forecast",
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=40, t=60, b=40)
            )
            st.plotly_chart(fig_pred, use_container_width=True)
