# This is a sample Python script.
from multiprocessing import Process
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import asyncio
import warnings

from binanceAPI import add_historical_klines
from DBrepository import DBObject as dbo
from predict_plot_db import metodname as metodname

def main():
    loop = asyncio.get_event_loop()
    try:
        db = dbo()
        db.createSchema()
        symbols = db.get_symbols()
        for symbol in symbols:
            add_historical_klines(symbol)
        for symbol in symbols:
            metodname(symbol)    
       
    except Exception as e:
        print('main exception: ')
        print(e)
        loop.stop()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
