from binance.client import Client
from binance import AsyncClient
from binance.exceptions import BinanceAPIException

import config
import pandas as pd
from binance import BinanceSocketManager

from DBApi import insert_data_frame
from DBrepository import DBObject as dbo

API_KEY = config.BINANCE['api_key']
SECRET_KEY = config.BINANCE['api_secret']
client = Client(API_KEY, SECRET_KEY, testnet=True)
symbol = 'BTCUSDT'
db = dbo()

def get_exchange_info():
    return client.get_exchange_info()
def get_account():
    return client.get_account()
def get_symbol_info(symbol):
    return client.get_symbol_info(symbol=symbol)
def get_symbol_ticker(symbol):
    return client.get_symbol_ticker(symbol=symbol)
async def get_multiplex_socket(multi):
    client_async = await AsyncClient.create()
    bm = BinanceSocketManager(client_async)
    return bm.multiplex_socket(multi)


async def get_socket_msg(symbol):
    socket = await get_socket(symbol)
    await socket.__aenter__()
    msg = await socket.recv()
    return msg


async def get_socket(symbol):
    bsm = BinanceSocketManager(client)
    socket = bsm.trade_socket(symbol)
    return socket

async def close_connection():
    await client.close_connection()

def add_historical_klines(symbol):
    df = getMinuteData(symbol, "1h", "90 days ago UTC")
    insert_data_frame(df, symbol)

def getMinuteData(symbol, interval, lookback):
    try:
        frame = pd.DataFrame(get_historical_klines(interval, lookback, symbol))
    except BinanceAPIException as e:
        print(e)
        frame = pd.DataFrame(get_historical_klines(interval, lookback, symbol))
    frame = frame.iloc[:, :6]
    frame.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    frame = frame.set_index('Time')
    frame.index = pd.to_datetime(frame.index, unit='ms')
    frame = frame.astype(float)
    return frame


def get_historical_klines(interval, lookback, symbol):
    return client.get_historical_klines(symbol, interval, lookback)


def getHistoricals(symbol, LT, ST):
    df = pd.DataFrame(client.get_historical_klines(symbol, '1d', str(LT) + 'days ago UTC', '1 day ago UTC'))
    closes = pd.DataFrame(df[4])
    closes.columns = ['Close']
    closes['ST'] = closes.Close.rolling(ST - 1).sum()
    closes['LT'] = closes.Close.rolling(LT - 1).sum()
    closes.dropna(inplace=True)
    return closes


def get_top_symbol():
    all_pairs = pd.DataFrame(client.get_ticker())
    relev = all_pairs[all_pairs.symbol.str.contains('USDT')]
    non_relev = relev[~((relev.symbol.str.contains('UP')) | (relev.symbol.str.contains('DOWN')))]
    top_symbol = non_relev[non_relev.priceChangePercent == non_relev.priceChangePercent.max()]
    top_symbol = top_symbol.symbol.values[0]
    return top_symbol


def create_order_sell(qty, symbol):
    order = client.create_order(symbol=symbol,
                                side='SELL', type='MARKET',
                                quantity=qty)
    return order


def create_order_buy(qty, symbol):
    order = client.create_order(symbol=symbol,
                                side='BUY', type='MARKET',
                                quantity=qty)
    return order
