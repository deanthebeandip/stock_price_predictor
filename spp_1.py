import pandas as pd
import yfinance as yf
import datetime
import time
import requests
import io
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers


# Helper Function Zone
def pd_tmstmp_to_str(t):
    return str(t)[:10]

def str_to_datetime(s):
  split = s.split('-')
  year, month, day = int(split[0]), int(split[1]), int(split[2])
  return datetime.datetime(year=year, month=month, day=day)







def df_to_windowed_df(dataframe, first_date_str, last_date_str, n=3):
    first_date = str_to_datetime(first_date_str)
    last_date  = str_to_datetime(last_date_str)

    target_date = first_date

    dates = []
    X, Y = [], []

    last_time = False
    while True:
        df_subset = dataframe.loc[:target_date].tail(n+1)

        if len(df_subset) != n+1:
            print(f'Error: Window of size {n} is too large for date {target_date}')
            return

        values = df_subset['Close'].to_numpy()
        x, y = values[:-1], values[-1]

        dates.append(target_date)
        X.append(x)
        Y.append(y)

        next_week = dataframe.loc[target_date:target_date+datetime.timedelta(days=7)]
        next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
        next_date_str = next_datetime_str.split('T')[0]
        year_month_day = next_date_str.split('-')
        year, month, day = year_month_day
        next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))

        if last_time:
            break

        target_date = next_date

        if target_date == last_date:
            last_time = True

    ret_df = pd.DataFrame({})
    ret_df['Target Date'] = dates
    
    X = np.array(X)
    for i in range(0, n):
        X[:, i]
        ret_df[f'Target-{n-i}'] = X[:, i]
    ret_df['Target'] = Y
    return ret_df

def windowed_df_to_date_X_y(windowed_dataframe):
    df_as_np = windowed_dataframe.to_numpy()
    dates = df_as_np[:, 0]
    middle_matrix = df_as_np[:, 1:-1]
    #   X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 6))
    X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))
    Y = df_as_np[:, -1]
    return dates, X.astype(np.float32), Y.astype(np.float32)



# evaluate model's stock prediction performance
def evaluate(ticker, start_date):
    today = str(datetime.date.today())
    tom = str(datetime.date.today()+ datetime.timedelta(days=1))
    trail = 10

    df = yf.download("VOO", #UPDATE
                # start="2010-09-09", 
                start= start_date, #UPDATE
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
    df['Date'] = df['Date'].apply(pd_tmstmp_to_str) # convert timestamp to string
    df['Date'] = df['Date'].apply(str_to_datetime) # convert string to datetime
    df.index = df.pop('Date')

    # plt.plot(df.index, df['Close'])
    # plt.show()

    # '2020-04-01',

    # Seems like windowed df is the problem, keeps getting stuck
    windowed_df = df_to_windowed_df(df, 
                                '2023-02-01', #UPDATE (make sure range makes sense, target date + n for this date)
                                today, #UPDATE (don't put tomorrow, it doesn't like that)
                                trail) #UPDATE
    
    print("Finished windowed df")
    dates, X, y = windowed_df_to_date_X_y(windowed_df)
    print("Finished windowed df to date X y")



    q_80 = int(len(dates) * .8)#UPDATE
    q_90 = int(len(dates) * .9)#UPDATE

    dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]

    dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
    dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]

    # plt.plot(dates_train, y_train)
    # plt.plot(dates_val, y_val)
    # plt.plot(dates_test, y_test)
    # plt.legend(['Train', 'Validation', 'Test'])
    # plt.show()


    model = Sequential([layers.Input((trail, 1)),#UPDATE
                    layers.LSTM(64),
                    layers.Dense(32, activation='relu'),#UPDATE
                    layers.Dense(32, activation='relu'),#UPDATE
                    layers.Dense(1)])

    model.compile(loss='mse', 
                optimizer=Adam(learning_rate=0.001),#UPDATE
                metrics=['mean_absolute_error'])#UPDATE

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=2)#UPDATE
    # model.fit(tf.expand_dims(X, axis=-1),y,epochs=10)
    train_predictions = model.predict(X_train).flatten()
    print(train_predictions)



    '''
    plt.plot(dates_train, train_predictions)
    plt.plot(dates_train, y_train)
    plt.legend(['Training Predictions', 'Training Observations'])
    # plt.show()

    val_predictions = model.predict(X_val).flatten()

    plt.plot(dates_val, val_predictions)
    plt.plot(dates_val, y_val)
    plt.legend(['Validation Predictions', 'Validation Observations'])
    # plt.show()

    figure(figsize=(20, 6), dpi=80)
    test_predictions = model.predict(X_test).flatten()

    plt.plot(dates_test, test_predictions)
    plt.plot(dates_test, y_test)
    plt.grid(axis="x")


    plt.legend(['Testing Predictions', 'Testing Observations'])
    plt.show()
    '''

def command_center():
    ticker = "VOO"
    start_date = "2010-09-09"
    evaluate(ticker, start_date)

    # try each "trail", return the score for each trail.
    # Try this for different models!





def main():
    command_center()



if __name__ == '__main__':
    main()