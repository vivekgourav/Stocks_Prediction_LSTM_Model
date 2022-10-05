#Importing Lib
import math
from tabulate import tabulate
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


#Get the stock quote
#Using Data Frame to read data from yahoo finance
df = web.DataReader('BAJFINANCE.NS', data_source='yahoo', start='2012-01-03', end='2022-10-02')
print(tabulate(df, headers = 'keys', tablefmt = 'psql'))

#Number of col and rows (Data)
size = df.shape
print(size)

#Plot the data of mentioned date with close proice vs date
plt.figure(figsize=(16,8))
plt.title('close Price Hist')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price INR', fontsize=18)
plt.show()

#Create datafram with close col

data = df.filter(['Close'])

#Conver df to a numpy array

dataset = data.values

#Get the number of rows to train the model
training_data_len = math.ceil(len(dataset) *0.8)
print(training_data_len)

#Scale the data for pre-processing before input to neural network
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

print(scaled_data)

#Create training data set
#Then create scaled training data set

training_data = scaled_data[0:training_data_len, :]

#Split the data into x_train and y_train

x_train = []
y_train = []

for i in range(60, len(training_data)):
    x_train.append(training_data[i-60:i,0])
    y_train.append(training_data[i,0])
    if i<=60:
        print(x_train)
        print(y_train)


#Convert the x_train y_train to numpy array

x_train, y_train = np.array(x_train), np.array(y_train)

#Reshape the data, to convert to 3d
x_train =np.reshape(x_train,(x_train.shape[0], x_train.shape[1], 1))

#Building the LSTM Model
model = Sequential()
model.add(LSTM(250, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(250, return_sequences=False))
model.add(Dense(45))
model.add(Dense(1))

#Compile

model.compile(optimizer='adam', loss='mean_squared_error')

#Train model
model.fit(x_train, y_train,batch_size=1, epochs=1)

#Create the testing data set
#Create new array containing scaled values from index
test_data = scaled_data[training_data_len -60: ,:]

#create dataset x and y_test
x_test= []
y_test = dataset[training_data_len:,:]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i,0])

x_test =np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))

#Get models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))

#Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('INR Price', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'val', 'predictions'])
plt.show()

share_quote = web.DataReader('BAJFINANCE.NS', data_source='yahoo', start='2012-01-03', end='2022-10-03')

new_df = share_quote.filter(['Close'])

#Get last 60 day closing and convert to array
last_60_days = new_df[-60:].values
last_60_days_scaled = scaler.transform(last_60_days)

X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)

#Reshape
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
pred_price = model.predict(X_test)

