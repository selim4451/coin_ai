import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from DBrepository import DBObject as dbo
import plotly.graph_objs as go
import ta

db = dbo()

def metodname(symbol):
    df = db.read_coin(symbol)
    df = df[['Close']].dropna()
    df.index = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq='1h')
    df.index.name = 'Time'

    # Teknik göstergeler
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    df['MACD'] = ta.trend.MACD(close=df['Close']).macd_diff()
    df['EMA_20'] = ta.trend.EMAIndicator(close=df['Close'], window=20).ema_indicator()
    df['EMA_50'] = ta.trend.EMAIndicator(close=df['Close'], window=50).ema_indicator()
    bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['bollinger_upper'] = bb.bollinger_hband()
    df['bollinger_lower'] = bb.bollinger_lband()
    df.dropna(inplace=True)

    # Ölçekleme
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
    timestamps = df.index[sequence_length + split:]

    # Modeli yükle
    model = load_model(f'models/{symbol}_lstm_model.h5')

    # Tahmin yap
    predicted_test = model.predict(X_test)
    y_test_real = scaler.inverse_transform(np.hstack([y_test.reshape(-1, 1), np.zeros((len(y_test), len(features)-1))]))[:, 0]
    predicted_test_real = scaler.inverse_transform(np.hstack([predicted_test, np.zeros((len(predicted_test), len(features)-1))]))[:, 0]

    # Gelecek tahmini
    last_input = scaled[-sequence_length:]
    future_predictions = []
    for _ in range(48):  # 2 gün 48 saat
        pred = model.predict(np.expand_dims(last_input, axis=0))[0, 0]
        new_input = np.append(last_input[1:], [[pred] + [0]*(len(features)-1)], axis=0)
        future_predictions.append(pred)
        last_input = new_input

    # Ters ölçekle
    future_pred_real = scaler.inverse_transform(np.hstack([np.array(future_predictions).reshape(-1, 1), np.zeros((48, len(features)-1))]))[:, 0]
    future_index = pd.date_range(start=df.index[-1] + pd.Timedelta(hours=1), periods=48, freq='H')

    # Grafik
    trace_real = go.Scatter(x=timestamps, y=y_test_real, mode='lines', name='Gerçek Fiyat')
    trace_pred = go.Scatter(x=timestamps, y=predicted_test_real, mode='lines', name='Tahmin (Geçmiş)')
    trace_future = go.Scatter(x=future_index, y=future_pred_real, mode='lines', name='Tahmin (Gelecek)', line=dict(dash='dash'))

    layout = go.Layout(
        title=symbol + " - Tahmin Sonuçları",
        xaxis=dict(title='Zaman'),
        yaxis=dict(title='Fiyat'),
        hovermode='x unified'
    )

    fig = go.Figure(data=[trace_real, trace_pred, trace_future], layout=layout)
    fig.show()
