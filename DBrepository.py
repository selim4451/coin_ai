import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import datetime as dt

import config


class DBObject():
    def __init__(self):
        self.conn = psycopg2.connect(
            database=config.DATABASE_CONFIG['database'],
            user=config.DATABASE_CONFIG['user'],
            password=config.DATABASE_CONFIG['password'],
            host=config.DATABASE_CONFIG['host'],
            port=config.DATABASE_CONFIG['port']
        )
        self.engine = create_engine("postgresql://postgres:12345@localhost/postgres")
        # establishing the connection
        self.conn.autocommit = True
        self.cursor = self.conn.cursor()

    def createSchema(self, schemanName='''binance'''):
        sql = '''CREATE SCHEMA IF NOT EXISTS %s''' % schemanName
        self.cursor.execute(sql)

    def insert_data_frame(self, df, symbol):
        df.to_sql(symbol, schema='binance', con=self.engine, if_exists='append',
                  index=False)

    def read_coin(self, symbol):
        return pd.read_sql('SELECT * FROM binance."' + symbol + '"', self.conn)

    def read_artanStoploss_coin(self, symbol, time):
        query = f'''SELECT * from binance."{symbol}" b WHERE \
            b."Time" >= '{time}' '''
        return pd.read_sql(query, self.conn)

    def get_symbols(self):
        return pd.read_sql(f'''SELECT table_name FROM information_schema.tables WHERE table_schema='binance' AND table_type='BASE TABLE' ''', self.conn).table_name.to_list()
    def get_symbolsStream(self):
        return pd.read_sql(f'''SELECT table_name FROM information_schema.tables WHERE table_schema='binances' AND table_type='BASE TABLE' ''', self.conn).table_name.to_list()

    def get_coin_from_db(self, symbol, lookback: int):
        now = dt.datetime.now() - dt.timedelta(hours=3)
        before = now - dt.timedelta(minutes=lookback)
        # qry_string = f"""SELECT * FROM '{symbol}' WHERE TIME >= '{before}'"""
        qry_string = f'''SELECT * from binance."{symbol}" b WHERE b."Time" >= '{before}' '''
        return pd.read_sql(qry_string, self.conn)



    def insert_value(self, timestmp, open_, close_, high, low, vol, amount, tablename):
        # CREATE A CURSOR USING THE CONNECTION OBJECT

        # EXECUTE THE INSERT QUERY
        self.cursor.execute(f'''
            INSERT INTO
                binance.{tablename}(time_, open_,close_,high,low,vol,amount) 
            VALUES
                ('{timestmp}', '{open_}', '{close_}', '{high}', '{low}', '{vol}', '{amount}')
        ''')
    # engine = sqlalchemy.create_engine('sqlite:///BTCUSDTstream.db')

    def connClose(self):
        self.cursor.close()
        self.conn.close()
