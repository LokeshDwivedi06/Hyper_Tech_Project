
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os
import plotly.graph_objects as go

# ---------- Page Config ----------
st.set_page_config(page_title="Stock Market Predictor", layout="wide")

# ---------- Title ----------
st.title("📈 Hyper_Tech Project")
st.markdown("Predict next day's stock prices using Deep Learning")

# ---------- Sidebar ----------
st.sidebar.header("Stock/Indices")

stock = st.sidebar.text_input("Enter Stock Symbol", "GOOG")

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2012-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2022-12-31"))

# ---------- Load Model ----------
@st.cache_resource
def load_my_model():
    model_path = os.path.join(os.getcwd(), "Stock Predictions Model.keras")
    return load_model(model_path)

model = load_my_model()

# ---------- Load Data ----------
data = yf.download(stock, start=start_date, end=end_date)

if data.empty:
    st.error("❌ No data found. Please check stock symbol.")
    st.stop()

# ---------- Show Data ----------
st.subheader(f"📊 Data for {stock}")
st.dataframe(data.tail())

# ---------- Moving Averages ----------
ma_50 = data.Close.rolling(50).mean()
ma_100 = data.Close.rolling(100).mean()
ma_200 = data.Close.rolling(200).mean()

# ---------- EMA ----------
ema_50 = data.Close.ewm(span=50, adjust=False).mean()
ema_100 = data.Close.ewm(span=100, adjust=False).mean()
ema_200 = data.Close.ewm(span=200, adjust=False).mean()

# ---------- Plot Price ----------
st.subheader("📉 Stock Trend")

fig1 = plt.figure(figsize=(12,6))
plt.plot(data.Close, label="Closing Price")
plt.plot(ma_50, label="MA50")
plt.plot(ma_100, label="MA100")
plt.plot(ma_200, label="MA200")
plt.legend()
st.pyplot(fig1)

# ---------- EMA PLOTS ----------
st.subheader("📊 Price vs 50 EMA")
fig_ema1 = plt.figure(figsize=(12,6))
plt.plot(data.Close)
plt.plot(ema_50)
st.pyplot(fig_ema1)

st.subheader("📊 Price vs 50 & 100 EMA")
fig_ema2 = plt.figure(figsize=(12,6))
plt.plot(data.Close)
plt.plot(ema_50)
plt.plot(ema_100)
st.pyplot(fig_ema2)

st.subheader("📊 Price vs 100 & 200 EMA")
fig_ema3 = plt.figure(figsize=(12,6))
plt.plot(data.Close)
plt.plot(ema_100)
plt.plot(ema_200)
st.pyplot(fig_ema3)

# ---------- BUY / SELL SIGNAL ----------
st.subheader("📉 Buy/Sell Signals (EMA Crossover)")

data['Signal'] = np.where(ema_50 > ema_100, 1, 0)

fig_signal = plt.figure(figsize=(12,6))
plt.plot(data.Close, label='Price')
plt.plot(ema_50, label='EMA 50')
plt.plot(ema_100, label='EMA 100')

buy = data[data['Signal'] == 1]
sell = data[data['Signal'] == 0]

plt.scatter(buy.index, buy.Close, marker='^', color='green', label='BUY')
plt.scatter(sell.index, sell.Close, marker='v', color='red', label='SELL')

plt.legend()
st.pyplot(fig_signal)

# ---------- Data Split ----------
data_train = pd.DataFrame(data.Close[0:int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80):])

# ---------- Scaling ----------
scaler = MinMaxScaler(feature_range=(0,1))
data_train_scaled = scaler.fit_transform(data_train)

# ---------- Prepare Test Data ----------
past_100 = data_train.tail(100)
data_test_full = pd.concat([past_100, data_test], ignore_index=True)
data_test_scaled = scaler.transform(data_test_full)

x = []
y = []

for i in range(100, data_test_scaled.shape[0]):
    x.append(data_test_scaled[i-100:i])
    y.append(data_test_scaled[i,0])

x, y = np.array(x), np.array(y)

# ---------- Prediction ----------
predictions = model.predict(x)

# ---------- Rescale ----------
scale_factor = 1 / scaler.scale_[0]
predictions = predictions * scale_factor
y = y * scale_factor

# ---------- Prediction Plot ----------
st.subheader("🔮 Prediction vs Actual")

fig2 = plt.figure(figsize=(12,6))
plt.plot(y, label="Actual Price")
plt.plot(predictions, label="Predicted Price")
plt.legend()
st.pyplot(fig2)

# ---------- Next Day Prediction ----------
st.subheader("🤖 Next Day Prediction")

last_100 = data_test_scaled[-100:]
last_100 = np.reshape(last_100, (1, 100, 1))

next_pred = model.predict(last_100)
next_pred = next_pred * scale_factor

st.success(f"📈 Predicted Next Day Price: {next_pred[0][0]:.2f}")

# ---------- Metrics ----------
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y, predictions))

st.subheader("📊 Model Performance")
st.write(f"**RMSE:** {rmse:.2f}")

# ---------- Extra ----------
if st.checkbox("Show Raw Data"):
    st.write(data)

if st.button("Show Summary"):
    st.write(data.describe())

st.success("✅ Prediction Completed Successfully!")