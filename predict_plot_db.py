import numpy as np
import pandas as pd
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from DBrepository import DBObject as dbo
import ta
db = dbo()
# 1. Binance'ten veri çek (1 saatlik verilerle)
def metodname(symbol):
    df = db.read_coin(symbol)
    df = df[['Close']].dropna()

    # Zaman ayarla
    df.index = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq='1h')
    df.index.name = 'Time'

    # 2. Teknik göstergeleri hesapla
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    df['MACD'] = ta.trend.MACD(close=df['Close']).macd_diff()
    df['EMA_20'] = ta.trend.EMAIndicator(close=df['Close'], window=20).ema_indicator()
    df['EMA_50'] = ta.trend.EMAIndicator(close=df['Close'], window=50).ema_indicator()
    bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['bollinger_upper'] = bb.bollinger_hband()
    df['bollinger_lower'] = bb.bollinger_lband()
    df.dropna(inplace=True)

    # 3. Ölçekle
    features = ['Close', 'RSI', 'MACD', 'EMA_20', 'EMA_50', 'bollinger_upper', 'bollinger_lower']
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])

    sequence_length = 120
    X, y = [], []
    for i in range(sequence_length, len(scaled)):
        X.append(scaled[i-sequence_length:i])
        y.append(scaled[i, 0])

    X, y = np.array(X), np.array(y)

    # Eğitim / Test bölme
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    timestamps = df.index[sequence_length + split:]

    # 4. Model kur
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test))

    # 5. Geçmiş veriler için tahmin
    predicted_test = model.predict(X_test)
    y_test_real = scaler.inverse_transform(np.hstack([y_test.reshape(-1, 1), np.zeros((len(y_test), len(features)-1))]))[:, 0]
    predicted_test_real = scaler.inverse_transform(np.hstack([predicted_test, np.zeros((len(predicted_test), len(features)-1))]))[:, 0]

    # 6. Gelecek 1 hafta (168 saat) için tahmin
    last_input = scaled[-sequence_length:]
    future_predictions = []
    for _ in range(48):
        pred = model.predict(np.expand_dims(last_input, axis=0))[0, 0]
        last_known_features = last_input[-1][1:]
        new_input = np.append(last_input[1:], [[pred] + [0]*(len(features)-1)], axis=0)  # Sadece fiyat tahmini yapıyoruz
        future_predictions.append(pred)
        last_input = new_input

    # Ters ölçekle
    future_pred_real = scaler.inverse_transform(np.hstack([np.array(future_predictions).reshape(-1, 1), np.zeros((48, len(features)-1))]))[:, 0]
    future_index = pd.date_range(start=df.index[-1] + pd.Timedelta(hours=1), periods=168, freq='H')

    # 7. Grafik çiz
    trace_real = go.Scatter(x=timestamps, y=y_test_real, mode='lines', name='Gerçek Fiyat')
    trace_pred = go.Scatter(x=timestamps, y=predicted_test_real, mode='lines', name='Tahmin (Geçmiş)')
    trace_future = go.Scatter(x=future_index, y=future_pred_real, mode='lines', name='Tahmin (Gelecek)', line=dict(dash='dash'))

    layout = go.Layout(
        title=symbol+" - LSTM ile Geçmiş ve 2 Günlük Gelecek Tahmini",
        xaxis=dict(title='Zaman'),
        yaxis=dict(title='Fiyat (USD)'),
        hovermode='x unified'
    )

    fig = go.Figure(data=[trace_real, trace_pred, trace_future], layout=layout)
    fig.show()
