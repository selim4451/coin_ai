import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import save_model
from tensorflow.keras.losses import MeanSquaredError
from DBrepository import DBObject as dbo
import ta
import os

db = dbo()

def train_model(symbol):
    df = db.read_coin(symbol)
    df = df[['Close']].dropna()
    df.index = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq='1h')
    df.index.name = 'Time'

    # Teknik göstergeleri hesapla
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    df['MACD'] = ta.trend.MACD(close=df['Close']).macd_diff()
    df['EMA_20'] = ta.trend.EMAIndicator(close=df['Close'], window=20).ema_indicator()
    df['EMA_50'] = ta.trend.EMAIndicator(close=df['Close'], window=50).ema_indicator()
    bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['bollinger_upper'] = bb.bollinger_hband()
    df['bollinger_lower'] = bb.bollinger_lband()
    df.dropna(inplace=True)

    # Özellikleri ölçekle
    features = ['Close', 'RSI', 'MACD', 'EMA_20', 'EMA_50', 'bollinger_upper', 'bollinger_lower']
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])

    sequence_length = 120
    X, y = [], []
    for i in range(sequence_length, len(scaled)):
        X.append(scaled[i-sequence_length:i])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)

    # Eğitim/test bölme
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # LSTM Modeli
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss=MeanSquaredError())
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    # Modeli kaydet
    os.makedirs("models", exist_ok=True)
    model.save(f"models/{symbol}_lstm_model.h5")



if __name__ == "__main__":
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    for symbol in symbols:
        print(f"{symbol} modeli eğitiliyor...")
        train_model(symbol)
        print(f"{symbol} modeli kaydedildi.\n")

