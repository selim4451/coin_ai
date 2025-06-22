from DBrepository import DBObject as dbo

db = dbo()


def insert_data_frame(df, symbol):
    db.insert_data_frame(df, symbol)


def get_coin_from_db(symbol, loocback):
    return db.get_coin_from_db(symbol=symbol, lookback=loocback)
