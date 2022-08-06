import talib as ta
import math
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout 


def get_data(ticker_name):
    #Retriev stock data for the last 2 years
    period = '1y'
    ticker = yf.Ticker(ticker_name)
    data = ticker.history(period=period)
    data = pd.DataFrame(data)
    data.to_csv(ticker_name + '.csv')
    return data


def add_indicators(data):
    # Add Bollinger Bands
    # data['BB_20'] = ta.BBANDS(data['Close'], 20, 2, 2, 0)[0]
    # data['BB_80'] = ta.BBANDS(data['Close'], 20, 2, 2, 0)[2]
    # Add EMA
    data['EMA_20'] = ta.EMA(data['Close'], 20)
    # data['EMA_100'] = ta.EMA(data['Close'], 50)
    # # data['EMA_100'] = ta.EMA(data['Close'], 100)
    # # Add MACD
    data['MACD_12_26'] = ta.MACD(data['Close'], 12, 26, 9)[0]
    # data['MACD_Signal_9'] = ta.MACD(data['Close'], 12, 26, 9)[1]
    # data['MACD_Hist_9'] = ta.MACD(data['Close'], 12, 26, 9)[2]
    # # Add RSI
    data['RSI_14'] = ta.RSI(data['Close'], 14)
    # # Add Stochastic Oscillator
    # data['Stoch_K'] = ta.STOCH(data['High'], data['Low'], data['Close'])[0]
    # data['Stoch_D'] = ta.STOCH(data['High'], data['Low'], data['Close'])[1]
    # # Add Williams %R
    # data['Williams_R'] = ta.WILLR(data['High'], data['Low'], data['Close'])
    # # Add Momentum
    data['Momentum'] = data['Close'].diff(1)
    # # Add ADX
    # data['ADX'] = ta.ADX(data['High'], data['Low'], data['Close'], 14)
    # # Add CCI
    # data['CCI'] = ta.CCI(data['High'], data['Low'], data['Close'], 20)
    # Add On-Balance Volume Indicator
    data['OBV'] = ta.OBV(data['Close'], data['Volume'])
    return data







def main():
    # ticker = input("Enter ticker to retrieve quote and TA: ")
    data = get_data('aapl')
    df_ta = add_indicators(data)
    tmp = df_ta.reset_index()
    close_dates = tmp['Date']
    print(close_dates)
    df_ta = df_ta.drop(columns=['Stock Splits', 'Dividends'])
    df_ta = df_ta.dropna()
    scaler = StandardScaler()
    scaler = scaler.fit(df_ta)
    df_scaled = scaler.transform(df_ta)

    train_x = []
    train_y = []

    n_future = 1 
    n_past = 14 

    for i in range(n_past, len(df_scaled) - n_future +1):
        train_x.append(df_scaled[i - n_past:i, 0:df_ta.shape[1]])
        train_y.append(df_scaled[i + n_future -1:i + n_future,0])

    train_x, train_y = np.array(train_x), np.array(train_y)
    print('TrainX shape: ', train_x.shape)
    print("TrainY shape: ", train_y.shape)



    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=True))
    model.add(LSTM(32, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(train_y.shape[1]))
    model.compile(optimizer='adam',loss='mse')
    model.summary()

    history = model.fit(train_x, train_y, epochs=10, batch_size=16, validation_split=0.1,verbose=1)

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()
    # n_future=90
    # forecast_period_dates = pd.date_range(list(train_x))




if __name__ == "__main__":
    main()
