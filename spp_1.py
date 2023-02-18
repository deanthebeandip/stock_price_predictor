import pandas as pd
import yfinance as yf
import datetime
import time
import requests
import io
import matplotlib.pyplot as plt


# Helper Function Zone
def pd_tmstmp_to_str(t):
    return str(t)[:10]

def str_to_datetime(s):
  split = s.split('-')
  year, month, day = int(split[0]), int(split[1]), int(split[2])
  return datetime.datetime(year=year, month=month, day=day)


# evaluate model's stock prediction performance
def evaluate():
    today = str(datetime.date.today())
    tom = str(datetime.date.today()+ datetime.timedelta(days=1))
    trail = 10

    df = yf.download("VOO", #UPDATE
                # start="2010-09-09", 
                start="2010-09-09", #UPDATE
                end=tom,#UPDATE
                # fetch data by interval (including intraday if period < 60 days)
                # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
                interval = "1d")
    # data['Close'].tail(n=5).to_list(
    # data["High"].plot()
    # plt.show()
    df.reset_index(inplace=True)
    df = df.rename(columns = {'index':'Date'})

    # df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    df = df[['Date', 'Close']]
    df['Date'] = df['Date'].apply(pd_tmstmp_to_str)
    df['Date'] = df['Date'].apply(str_to_datetime)
    df.index = df.pop('Date')

    plt.plot(df.index, df['Close'])
    plt.show()










def main():

    evaluate()



if __name__ == '__main__':
    main()