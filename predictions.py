# -*- coding: utf-8 -*-
"""
Created on Fri May  3 15:17:06 2024

@author: franc
"""

# Import needed libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from tensorflow.keras.models import Sequential
from keras.layers import MaxPooling1D, Conv1D, Flatten, Dense

# IMPORT DATA
# Import .csv data file.
path = ''.join(("C:/Users/franc/OneDrive/Escritorio/Stock Prediction",
                "/AAPL_daily_data.csv"))
data = pd.read_csv(path, parse_dates=['Date'])

# Set date as dataframe index.
data = data.set_index('Date')

# EXPLORATORY DATA ANALYSIS
# View dimensions of the dataset.
data.shape

# Preview the dataset.
data.head()

# View summary of the dataset.
data.info()

# Check for missing values.
data.isnull().sum()

# Summary statistics of all columns.
data.describe(include='all')

# Plot the Adjusted Close Price.
plt.plot(data['Adj Close'])
plt.xticks(rotation=90)
plt.title('Precio de cierre ajustado AAPL')
plt.show()

# Plot the distributions.
fig = plt.figure(figsize=(10,7))
gs = gridspec.GridSpec(2, 3, figure=fig)

# For the open price.
ax1 = fig.add_subplot(gs[0,0])
ax1.hist(data['Open'], color='lightgreen', ec='black', bins=15)
ax1.set_title('Open Price')

# For the max price.
ax2 = fig.add_subplot(gs[0,1])
ax2.hist(data['High'], color='lightgreen', ec='black', bins=15)
ax2.set_title('Max Price')

# For the min price.
ax3 = fig.add_subplot(gs[0,2])
ax3.hist(data['Low'], color='lightgreen', ec='black', bins=15)
ax3.set_title('Min Price')

# For the adjusted close price.
ax4 = fig.add_subplot(gs[1,0])
ax4.hist(data['Adj Close'], color='lightgreen', ec='black', bins=15)
ax4.set_title('Adjusted Close Price')

# For the volume.
ax5 = fig.add_subplot(gs[1,1])
ax5.hist(data['Volume'], color='lightgreen', ec='black', bins=15)
ax5.set_title('Volume')
plt.show()

# CREATE A VARIABLE FOR THE OBJECTIVE VARIABLE.
obj_var = [data['Close'][i+1] for i in range(len(data)-1)]

# Drop the last row.
data.drop(data.tail(1).index, inplace=True)

# Append the objective variable to the data.
data['Obj Var'] = obj_var

# SPLIT THE SAMPLE INTO TRAIN, VALIDATION AND TEST
def split_sample(df):
    features = df.drop(columns=['Obj Var'], axis=1)
    target = df['Obj Var']
    
    data_len = len(data)
    
    train_split = int(data_len * 0.8)
    val_split = train_split + int(data_len * 0.1)
    
    # Split target and features into train, validation and test samples.
    X_train, X_val, X_test = features[:train_split], features[train_split:val_split], features[val_split:]
    Y_train, Y_val, Y_test = target[:train_split], target[train_split:val_split], target[val_split:]
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test

X_train, X_val, X_test, Y_train, Y_val, Y_test = split_sample(data)

# FIT THE LINEAR MODEL.
lr = LinearRegression()
lr.fit(X_train, Y_train)

# Predict for the test dataset.
Y_train_pred = pd.DataFrame(lr.predict(X_train))
Y_train_pred.index = Y_train.index
Y_val_pred = pd.DataFrame(lr.predict(X_val))
Y_val_pred.index = Y_val.index
Y_test_pred = pd.DataFrame(lr.predict(X_test))
Y_test_pred.index = Y_test.index

# Evaluate the model.
print(f"Performance (R^2): {lr.score(X_train, Y_train)}")
mse_lr = metrics.mean_squared_error(Y_test, Y_test_pred)
rmse_lr = np.sqrt(mse_lr)
mae_lr = metrics.mean_absolute_error(Y_test, Y_test_pred)

# Plot predicted vs. actual Close prices.
plt.plot(Y_test, color='green', label='Test data')
plt.plot(Y_test_pred, color='red', label='Predcited values')
plt.legend()
plt.title('Actual vs. Predicted Close Prices. AAPL. Linear model')
plt.xticks(rotation=90)
plt.show()

# TRAIN CONVOLUTIONAL NEURAL NETWORK.
# Define a function to split the multivariate sequence into samples.
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # Find the end of this pattern.
        end_ix = i + n_steps
        # Check if we are beyond the dataset.
        if end_ix > len(sequences):
            break
        # Gather input and output parts of the pattern.
        seq_x, seq_y = sequences.iloc[i:end_ix, :-1], sequences.iloc[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
        
    return np.array(X), np.array(y)

# Choose a number of time steps and get the number of features.
data_conv = data.drop('Volume', axis=1)
n_steps = 3
n_features = len(data_conv.columns)-1

# Convert the data into input/output.
X, y = split_sequences(data_conv, n_steps)

# Define the model.
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', 
                 input_shape=(n_steps, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Split the samples into train, validation and test.
train_split = int(len(X) * 0.8)
val_split = train_split + int(len(X) * 0.1)
dates_train, X_train, y_train = data.index[:train_split], X[:train_split], y[:train_split]
dates_val, X_val, y_val = data.index[train_split:val_split], X[train_split:val_split], y[train_split:val_split]
dates_test, X_test, y_test = data.index[val_split:-2], X[val_split:], y[val_split:]

# Fit the model.
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)

# Get predictions for the test data set.
y_test_pred = model.predict(X_test).flatten()

# Create a data frame for the predictions.
y_test_pred = pd.DataFrame(y_test_pred, index=dates_test)

# Graph the actual vs. predicted data.
plt.plot(dates_test, y_test_pred, label='Testing Predictions', color='red')
plt.plot(dates_test, y_test, label='Testing Observations', color='green')
plt.legend()
plt.title('Actual vs. Predicted Close Prices. AAPL. Convolutional Neural Network')
plt.xticks(rotation=90)
plt.show()

# Evaluate the model. 
mse_conv = metrics.mean_squared_error(y_test, y_test_pred)
mae_conv = metrics.mean_absolute_error(y_test, y_test_pred)
rmse_conv = np.sqrt(mse_conv)