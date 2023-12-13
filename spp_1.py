import pandas as pd
import yfinance as yf
import datetime
from datetime import date

import time
import requests
import io
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np

from tensorflow import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras import layers
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# Helper Function Zone
def pd_tmstmp_to_str(t):
    return str(t)[:10]

def str_to_datetime(s):
  split = s.split('-')
  year, month, day = int(split[0]), int(split[1]), int(split[2])
  return datetime.datetime(year=year, month=month, day=day)


# YT Functions to create window Dataframes.
def df_to_windowed_df(dataframe, first_date_str, last_date_str, n, var_1d):
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

        values = df_subset[var_1d].to_numpy()
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




def download_price(ticker, start_date, var_1D): #Download stock price data
    tom = str(datetime.date.today()+ datetime.timedelta(days=1))
    df = yf.download(ticker, 
                start= start_date, 
                end=tom,
                interval = "1d")# valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo

    df.reset_index(inplace=True)
    df = df.rename(columns = {'index':'Date'})
    # df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    df = df[['Date', var_1D]]
    df['Date'] = df['Date'].apply(pd_tmstmp_to_str) # convert timestamp to string
    df['Date'] = df['Date'].apply(str_to_datetime) # convert string to datetime
    df.index = df.pop('Date')

    return df

def create_window(stock_dataframe, runway, window_start_date, var_1d):
    window_start_date = str(str_to_datetime(window_start_date)+ datetime.timedelta(days=runway))[:10]
    end_date = str(datetime.date.today())
    wday = date.today().weekday()
    if wday == 5 or wday == 6: end_date = str(datetime.date.today()- datetime.timedelta(days=(wday-4)))

    windowed_df = df_to_windowed_df(stock_dataframe, 
                                window_start_date, #UPDATE (make sure range makes sense, target date + n for this date)
                                end_date, # ONLY ACCEPTS TRADING DAYS
                                runway,
                                var_1d) 
    
    print("Finished windowed df")
    dates, X, y = windowed_df_to_date_X_y(windowed_df)
    print("Finished windowed df to date X y")

    return dates, X, y

# Parse the data into train, validation, and test sets
def train_data(dates, X, y, train_end, val_end):
    q_80 = int(len(dates) * train_end)
    q_90 = int(len(dates) * val_end)
    return dates[:q_80], X[:q_80], y[:q_80]
def val_data(dates, X, y, train_end, val_end):
    q_80 = int(len(dates) * train_end)
    q_90 = int(len(dates) * val_end)
    return dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
def test_data(dates, X, y, train_end, val_end):
    q_80 = int(len(dates) * train_end)
    q_90 = int(len(dates) * val_end)
    return dates[q_90:], X[q_90:], y[q_90:]


def model_training_room(X_train, y_train, X_val, y_val, trail, epochs_input, avt_input, lstm_layers, lstm_d1_layers, lstm_d2_layers, loss_input, lr, metrics_input):
    model = Sequential([layers.Input((trail, 1)),
                layers.LSTM(lstm_layers),
                layers.Dense(lstm_d1_layers, activation= avt_input),#UPDATE
                layers.Dense(lstm_d2_layers, activation= avt_input),#UPDATE
                layers.Dense(1)])

    model.compile(loss=loss_input, 
                optimizer=Adam(learning_rate=lr),#UPDATE
                metrics=[metrics_input])#UPDATE

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs_input)
    # model.fit(tf.expand_dims(X, axis=-1),y,epochs=10)
    return model
    
def plot_train_predictions(model, dates_train, X_train, y_train):
    train_predictions = model.predict(X_train).flatten()
    print_train = 0
    if print_train:
        plt.plot(dates_train, train_predictions)
        plt.plot(dates_train, y_train)
        plt.legend(['Training Predictions', 'Training Observations'])
        plt.show()

def plot_val_predictions(model, dates_val, X_val, y_val):
    val_predictions = model.predict(X_val).flatten()
    print_val = 0
    if print_val:
        plt.plot(dates_val, val_predictions)
        plt.plot(dates_val, y_val)
        plt.legend(['Validation Predictions', 'Validation Observations'])
        plt.show()


def model_evaluation(pred, obs):
    scores = []
    slopes = []

    for i in range(len(pred)):
        scores.append(abs((pred[i]-obs[i])/obs[i]))

        # Work on a metric to penalize differences in price slope
        # eg slope is actually 2, but predicted 1.25 or something..
        #         # if i > 0:
        #     slopes.append(abs((pred[i]-pred[i-1])/pred[i-1]))


        # Work on a metric to penalize differences in price 
        #         # if i > 0:
        #     slopes.append(abs((pred[i]-pred[i-1])/pred[i-1]))


    return 100*sum(scores)/len(scores)

def plot_test_predictions(model, dates_test, X_test, y_test):
    test_predictions = model.predict(X_test).flatten()
    print_test = 0
    if print_test:
        figure(figsize=(20, 6), dpi=80)
        plt.plot(dates_test, test_predictions)
        plt.plot(dates_test, y_test)
        plt.xticks(dates_test)
        plt.grid(axis = 'x')
        plt.legend(['Testing Predictions', 'Testing Observations'])
        plt.show()


    # Calculate the scoring metrics here!
    scores = model_evaluation(test_predictions, y_test)

    return scores

# The Pilots Cabin
def score_model_api(ticker, stock_genesis_date, window_start_date, runway, 
                   var_1d, train_end, val_end, epochs, activation, lstm_layers, lstm_d1_layers, lstm_d2_layers, loss_input, lr, metrics_input): # Fill in parameters in the beginning

    # Grab Stock Price DataFrame
    stock_df = download_price(ticker, stock_genesis_date, var_1d)
    # Create Windowed DataFrame (using YT method)
    dates, X, y = create_window(stock_df, runway, window_start_date, var_1d)
    # Split into train, val, test
    dates_train, X_train, y_train = train_data(dates, X, y, train_end, val_end)
    dates_val, X_val, y_val = val_data(dates, X, y, train_end, val_end)
    dates_test, X_test, y_test = test_data(dates, X, y, train_end, val_end)
    # Train the model with given paramters
    model = model_training_room(X_train, y_train, X_val, y_val, runway, epochs, activation, 
                                lstm_layers, lstm_d1_layers, lstm_d2_layers, loss_input, 
                                lr, metrics_input)


    #Use the model to evaluate the test data
    plot_train_predictions(model, dates_train, X_train, y_train)
    plot_val_predictions(model, dates_val, X_val, y_val)
    model_scores = plot_test_predictions(model, dates_test, X_test, y_test)
    return model_scores

def get_stock_data(ticker, start_date, end_date):
    try:
        # Download historical data
        stock_data = yf.download(ticker, start=start_date, end=end_date)

        # Extracting the 'Close' prices
        df = stock_data['Close'].to_frame()

        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None





def get_stock_prices(symbol, param, start_date, end_date):
    try:
        # Download historical stock data
        stock_data = yf.download(symbol, start=start_date, end=end_date)

        # Extract the closing prices
        stock_prices = stock_data[param]

        return stock_prices

    except Exception as e:
        print(f"Error: {e}")
        return None




def main():
    # Example usage:
    stock_symbol = "VOO"  # Replace with the desired stock symbol
    param = 'Close'
    start_date = "2022-01-01"  # Replace with the start date
    end_date = "2023-12-13"  # Replace with the end date


    prices = get_stock_prices(stock_symbol, param, start_date, end_date)

    if prices is not None:
        print(f"Stock prices for {stock_symbol} from {start_date} to {end_date}:\n")
        print(prices)
    else:
        print(f"Failed to retrieve stock prices for {stock_symbol}")


# Next step:
# Create an experiment database that contains all the paramters as dimensions
# Then add the scores as metrics!

# Then I could see which models are giving me the most accurate models


if __name__ == '__main__':
    main()



#old main:

    """
    ticker = "AAPL"  # Example stock ticker (Apple Inc.)
    stock_genesis_date = '2010-09-09'
    var_1D = 'Close'
    stock_df = download_price(ticker, stock_genesis_date, var_1D)
    print(stock_df)
    """


    ''' 230201 Stuff
    # Stock Input Parameters
    ticker = "VOO"                      # What stock u want?
    stock_genesis_date = '2010-09-09'   # When did the stock start trading?
    window_start_date = '2020-06-02'    # Model Training start date
    runway = 7                         # How many days to guess tmrw's price?
    var_1d = 'Low'    #'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'
    
    # Model Input Parameters
    train_end = .75                      #Where to stop training data
    val_end = .85                        #Where to stop validation data
    epochs = [50, 100]
    #epochs = [50, 100,500, 1000]
    activation = 'relu'
    lstm_layers = 64
    lstm_d1_layers = 32
    lstm_d2_layers = 32
    loss_input = 'mse'
    lr = 0.001
    metrics_input = 'mean_absolute_error'

    score_list = {}
    for epoch in epochs:

        score = score_model_api(ticker, stock_genesis_date, window_start_date, 
                    runway, var_1d, train_end, val_end, epoch, activation, lstm_layers, lstm_d1_layers, lstm_d2_layers, loss_input, lr, metrics_input)
        print("%i avg. error is %f" % (epoch, score))
        score_list[epoch] = score

    print(score_list)
    '''