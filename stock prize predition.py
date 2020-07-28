# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 12:18:15 2020

@author: D Pranay kumar naidu
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import math

df = pdr.DataReader('AAPL', data_source = 'yahoo', start = '2013-01-02', end = '2020-05-29')
training_set = math.ceil(len(df) * .8)
print(training_set)

training_set = df.iloc[:1493, 2:3].values
test_set =  df.iloc[1493:, 2:3].values

#normailising the training set
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)
 
# splitting the data into X_train, y_train
X_train = []
y_train = []
for i in range(70, 1493):
    X_train.append(training_set_scaled[i-70:i, 0])
    y_train.append(training_set_scaled[i, 0])

X_train = np.array(X_train)
y_train = np.array(y_train)

# input shape for the RNN

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# importing the tensorflow and keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout


regressor = Sequential()
regressor.add(LSTM(units = 60, return_sequences = True, 
                   input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True ))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True ))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50 ))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))


regressor.compile(optimizer = 'Adam', loss = 'mean_squared_error')
regressor.fit(X_train, y_train, epochs = 70 , batch_size = 32)

df_new = df['Open']
inputs = df_new[len(df_new) - len(test_set) - 70:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []
for i in range(70, 442):
    X_test.append(inputs[i-70:i, 0])
X_test = np.array(X_test)    
X_test = np.reshape(X_test , (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_prize = regressor.predict(X_test)
predicted_stock_prize = sc.inverse_transform(predicted_stock_prize)

plt.plot(test_set, color ='red',label = 'real stock price')
plt.plot(predicted_stock_prize, color = 'blue', label = 'predicted stock price')
plt.title('APPLE stock price predictoin')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
plt.show()




























