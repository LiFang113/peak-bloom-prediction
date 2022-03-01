#!/usr/bin/env python3
#predict cherry blossom DOY using Univariate Time Series Deep Learning
#by Li Fang (fangli113@gmail.com)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import dl_utility

from pandas import Series
import matplotlib as mpl
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

import tensorflow as tf
from tensorflow import keras 

import warnings
warnings.filterwarnings("ignore")

#### Function to rescale whatever transformations we have done, 
#### this will be used to retransform the values to original values after forecast.
def reverse_transform(arr):
    
    #print(arr.shape)
    arr = arr.reshape(-1,1)
    
    ## First reverse the minmax scaling
    arr_inv_normal = scaler.inverse_transform(arr)
    #print(arr_inv_normal.shape)
    
    ## Reverse the log transformation 
    arr_reverse = np.exp(arr_inv_normal )
    #print(arr_reverse.shape)
    return (arr_reverse)

np.set_printoptions(suppress=True) 

###cherry blossom doy for 69 locations over 69 years (1953-2021)
fn = "univariate_inputs.csv"

df = pd.read_csv(fn)
print("df orig shape", df.shape)

new_df = df.drop(['Location'], axis=1)
print(new_df.head(10))

###Change the column names from dates to time steps
new_df.columns = [ i for i,x in enumerate(new_df.columns)]
#print(new_df.head(10))

###convert all the data in to 3-D array 
#inputs array: [ batch-size, Time-Steps, Number of elements in a single timestep ]

max_steps = 80
master_series_stations = []
master_series = []

for i in range(len(new_df)):
    
    myseries = np.array(new_df.iloc[i][0:max_steps])
    master_series.append(myseries)
    
master_series = np.array(master_series) 
master_series = master_series[..., np.newaxis].astype(np.float32)

print(master_series.shape)

###Checking and Removing Outliers
master_series_values = master_series.reshape(-1,1)

print(np.percentile(master_series_values, [1, 5, 25, 50, 75, 95, 98, 99]))
print(master_series_values.mean())
print(master_series_values.max())
####output: [ 31.   86.   94.  100.  115.  134.  140.  144.4]
#            103.55997
#            162.0

### Shuffle the data set
np.random.seed(42)
np.random.shuffle(master_series)

###Scaling of Time Series Data
log_master_series = []

for i in range(master_series.shape[0]):
    
    temp_series = np.array(np.log(master_series[i][0:]))
    log_master_series.append(temp_series)
    
log_master_series = np.array(log_master_series) 

###split the data in to train, test and validation sets
np.random.seed(42)

n_steps = 58
num_days_predicted = 11

X_train, y_train = log_master_series[:55, :n_steps], log_master_series[:55, n_steps:n_steps+num_days_predicted:, 0]
X_valid, y_valid = log_master_series[55:69, :n_steps], log_master_series[55:69, n_steps:n_steps+num_days_predicted:, 0]

print('train shape:', X_train.shape, y_train.shape)
print('valid shape:', X_valid.shape, y_valid.shape)
print('x_train before scale:', X_train[0, 0:5])

###normalize the data set to bring the values between 0 and 1
#MINMAX NORMALIZATION
# Get X_train values reshaped in to 2D for scaler
X_train_values = X_train.reshape(-1, 1)

print(X_train_values.shape)
print(min(X_train_values))
print(max(X_train_values))

hisfig = plt.subplots()
sns.distplot(X_train_values)
plt.savefig('Density_scaled.png')

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

### Fit on X_train_values and transform 
X_train_normalized = scaler.fit_transform(X_train_values)
#print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))

### Transform y_train
y_train_normalized = scaler.transform(y_train.reshape(-1,1))

### Transform X_Valid
X_valid_values = X_valid.reshape(-1, 1)
X_valid_normalized = scaler.transform(X_valid_values)

### Transform y_valid
y_valid_normalized = scaler.transform(y_valid.reshape(-1,1))
y_alltrain_normalized = scaler.transform(y_alltrain.reshape(-1,1))

# Reshape normalized values back to 3-D
X_train = X_train_normalized.reshape(X_train.shape[0] , X_train.shape[1] , X_train.shape[2])
X_valid = X_valid_normalized.reshape(X_valid.shape[0] ,X_valid.shape[1] , X_valid.shape[2])

y_train = y_train_normalized.reshape(y_train.shape[0], y_train.shape[1] )
y_valid = y_valid_normalized.reshape(y_valid.shape[0], y_valid.shape[1] )

### Just a sanity check if any training data is Inf due to normalization..
### that will lead to exploding gradients issue while training
#for i in range(55):
#   for j in range(66):
#      if (np.isfinite(X_train[i,j,0]) == False):
#        print("Inf value at ", i)

###check reverse_transform function
#print('x_train after scale:', X_train[0, 0:5])
#print('reversed x_train',reverse_transform(X_train[0,0:5]))
#print('revesed y_valid', reverse_transform(y_valid[0:5]))
#print('master [55:60]', master_series[55:60, n_steps, 0])

######################################
###Deep RNN
np.random.seed(42)
tf.random.set_seed(42)

d_rnn_model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.Dropout(rate=0.2),   ### Otherwise validation loss fluctuates too much
    keras.layers.SimpleRNN(20),
    keras.layers.Dense(num_days_predicted)       ### Note dense layer with one output (One day prediction)
])

optimizer = keras.optimizers.Adam(lr=0.0001)

d_rnn_model.compile(loss="mse", optimizer=optimizer)

d_rnn_model.summary()
print()

#### Early stop the training if there is no improvement in val_loss for 5 epochs. Save the best model based on val_loss.
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
mc = keras.callbacks.ModelCheckpoint('best_model_drnn.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

history_drnn = d_rnn_model.fit(X_train, y_train, epochs=100, batch_size=32,
                    validation_data=(X_valid, y_valid), callbacks=[early_stopping, mc])

pngfl = "DeepRNN_LearningCurve.png"
dl_utility.plot_learning_curves(history_drnn.history["loss"][1:], history_drnn.history["val_loss"][1:], pngfl)

d_rnn_model = keras.models.load_model('best_model_drnn.h5')
print("Deep RNN MSE = ", round(d_rnn_model.evaluate(X_valid, y_valid), 7))

y_pred_drnn = d_rnn_model.predict(X_valid)

##############################################
###LSTM Mode
np.random.seed(42)
tf.random.set_seed(42)

lstm_mult_model1 = keras.models.Sequential([
    keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.LSTM(20, return_sequences=True), ####
    keras.layers.Dropout(rate=0.2), ####
    keras.layers.LSTM(20),
    keras.layers.Dense(num_days_predicted)   #### Note - Number of Output = Number of days predicted.
])

optimizer = keras.optimizers.Adam(lr=0.0005)

lstm_mult_model1.compile(loss="mse", optimizer=optimizer)
lstm_mult_model1.summary()

#### Early stop the training if there is no improvement in val_loss for 5 epochs. Save the best model based on val_loss.
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
mc = keras.callbacks.ModelCheckpoint('best_model_lstm.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

history_mult_lstm1 = lstm_mult_model1.fit(X_train, y_train, epochs=100, batch_size=32,
                    validation_data=(X_valid, y_valid), callbacks=[early_stopping, mc])

lstm_mult_model1 = keras.models.load_model('best_model_lstm.h5')
lstm_mult_model1.evaluate(X_valid, y_valid)

pngfl = "LSTM_MultiDay_Predict_LearningCurve.png"
dl_utility.plot_learning_curves(history_mult_lstm1.history["loss"][1:], history_mult_lstm1.history["val_loss"][1:],pngfl)

y_pred_lstm = lstm_mult_model1.predict(X_valid)


#########################################
#plot certain site for validation period
for sample in range(69):
    stationID = df['Location'][sample]
#    print("stationID = ", stationID)
    lag=n_steps - 1
    pngfl = "LSTM_MultDay_forecast_" + stationID + ".png"
    dl_utility.plot_multiforecasts(reverse_transform(X_alltrain[sample, n_steps - lag : , 0]), \
                         reverse_transform(y_train[sample]), \
                         reverse_transform(y_pred_drnn[sample]), \
                         reverse_transform(y_pred_lstm[sample]), \
                         station = stationID, n_steps = n_steps, \
                         lag = lag,pngfl=pngfl, num_days_predicted=num_days_predicted, \
                         x_label = "years from 1953", \
                         y_label = "CherryBlossomDOY")






