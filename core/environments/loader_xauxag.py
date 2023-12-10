import os
import pandas as pd


class Loader:
    def __init__(self):
        pass

    def read_data(self, symbol):
        df = pd.DataFrame()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        data_folder = os.path.join(dir_path + f"/data/{symbol}")
        symbol_candlesticks = os.listdir(data_folder)

        for symbol_candlestick in symbol_candlesticks:
            if symbol_candlestick.endswith(".csv"):
                data = pd.read_csv(os.path.join(dir_path + f"/data/{symbol}/{symbol_candlestick}"), sep=",")
                # data.index = pd.to_datetime(data['Gmt time'], unit='s')
                # Gmt time to timestamp
                data.index = pd.to_datetime(data['Gmt time'], format='%d.%m.%Y %H:%M:%S.%f')
                data = data.drop(['Gmt time'], axis=1)


                df = pd.concat([df, pd.DataFrame(data)], ignore_index=False)
        return df

    def add_indicators(self, df):
        df["ratio"] = df["XAUUSD"] / df["XAGUSD"]

        return df

    def load(self, start_date='2019-01-01', end_date='2019-12-31'):
        df_XAU = self.read_data("XAUUSD")
        df_XAG = self.read_data("XAGUSD")

        df_XAU = df_XAU.loc[(df_XAU.index >= start_date) & (df_XAU.index <= end_date)]
        df_XAU["XAUUSD"] = df_XAU["Close"]
        df_XAU = df_XAU.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1)
        df_XAU = df_XAU.resample("5min").agg({"XAUUSD": "last"}).dropna()

        df_XAG = df_XAG.loc[(df_XAG.index >= start_date) & (df_XAG.index <= end_date)]
        df_XAG["XAGUSD"] = df_XAG["Close"]
        df_XAG = df_XAG.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1)
        df_XAG = df_XAG.resample("5min").agg({"XAGUSD": "last"}).dropna()

        df = pd.concat([df_XAU, df_XAG], axis=1)
        df = df.ffill()

        df = self.add_indicators(df)
        df = df.dropna()
        print(df.head(10))
        return df

if __name__ == '__main__':
    loader = Loader()
    df = loader.load(start_date='2019-01-01', end_date='2019-12-31')
    print(df)