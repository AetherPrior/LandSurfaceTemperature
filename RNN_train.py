import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras import Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from osgeo import gdal
import os

# RNN fails to detect anything more than 1 timestep further
# Require further model


time_series_list = []
for f in os.listdir("Dataset/interpolated"):
    df = pd.read_csv(
        os.path.join("Dataset/interpolated", f))
    df.columns = ["date", "lat", "long", "temp"]
    df2 = df["temp"].iloc[0:-1]
    # print(df2)
    df = df.drop(df.head(1).index).reset_index()

    df['temp_1'] = df2
    df = df.reindex(columns=["date", "lat", "long", "temp_1", "temp"])
    # print(df.tail())
    time_series_list.append(df.to_numpy())
time_series_list = np.array(time_series_list)

mean_scale = np.mean(time_series_list[:, :, 4])
std_scale = np.std(time_series_list[:, :, 4])

for i in range(1, time_series_list.shape[2]):
    time_series_list[:, :, i] -= np.mean(time_series_list[:, :, i])
    time_series_list[:, :, i] /= np.std(time_series_list[:, :, i])

    """normalize
    time_series_list[:, :, i] -= np.min(time_series_list[:, :, i])
    time_series_list[:, :, i] /= (np.max(time_series_list[:, :, i]) -
                                  np.min(time_series_list[:, :, i]))
    """

X, y = time_series_list[:, :, 0:4], time_series_list[:, :, 4]
X_train, X_test, y_train, y_test = X[:, :int(time_series_list.shape[1]*0.7), :], X[:, int(time_series_list.shape[1]*0.7):int(
    time_series_list.shape[1]), :], y[:, :int(time_series_list.shape[1]*0.7)], y[:, int(time_series_list.shape[1]*0.7):int(time_series_list.shape[1])]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

model = Sequential()
model.add(Dense(12, input_shape=(None, 4)))
model.add(LSTM(12, input_shape=(X_train.shape[1], 4), return_sequences=True))
model.add(Dense(1))

model.compile(optimizer='adam',
              loss='mae')
model.summary()
if(os.path.isfile("./ckpt.h5")):
    model = tf.keras.models.load_model("./ckpt.h5")
callback = tf.keras.callbacks.EarlyStopping(patience=10)

# history = model.fit(X_train, y_train, epochs=4,
#                    shuffle=True,  callbacks=[callback], validation_split=0.2)

model.evaluate(X, y)
model.save("./ckpt.h5")
y_hat = model.predict(tf.expand_dims(
    X[0], axis=0))*(std_scale)+mean_scale
print(y_hat)

time = np.linspace(1, len(y[0]), len(y[0]))
'''
ax1 = plt.subplot(2, 1, 1)
ax1.plot(time[5:], np.squeeze(y_hat)[5:])
ax2 = plt.subplot(2, 1, 2)
'''

line1, = plt.plot(time[len(time)-160:], np.squeeze(y[0])
                  [len(time)-160:]*(std_scale)+mean_scale, c='orange')
line2, = plt.plot(time[len(time)-160:], np.squeeze(y_hat)
                  [len(time)-160:], c='blue')
plt.xlabel("Days")
plt.ylabel("Temperature (C)")
plt.legend([line1, line2], ["Target", "LSTM Predictions"])
plt.savefig("plots/LSTM_Cell.png")
plt.show()
plt.clf()


def model_serious_predict(model, data, final_y, final_index, req_index, norm=False):
    """
    data: index lat long temp [temp_prev] (can be zero)
    """
    if norm:
        for i in range(data.shape[2]):
            data[:, :, i] -= np.min(data[:, :, i])
            data[:, :, i] /= (np.max(data[:, :, i]) -
                              np.min(data[:, :, i]))
    predictions = []
    # data = np.append(data, [[[0]]], axis=2)
    dat = data.copy()
    print(dat.shape, data.shape)
    # dat.setflag(write=1)
    while final_index < req_index:
        # Update X
        dat[:, :, 3] = final_y
        dat[:, :, 0] += 1
        final_y = np.squeeze(model.predict(dat))
        predictions.append(final_y)
        final_index += 1
    return predictions


final_index = X.shape[1]
req_index = final_index + 10
data = np.expand_dims(tf.expand_dims(X[0, X.shape[1]-1, 0:4], axis=0), axis=1)
print(data)
final_y = y[0, y.shape[1]-1]


predictions = model_serious_predict(
    model, data, final_y, final_index, req_index)
plt.clf()
print(predictions, mean_scale, std_scale)
a = np.linspace(1, len(predictions), len(predictions))
plt.plot(a, np.multiply(predictions, (std_scale))+mean_scale)
plt.xlabel("Predictions since the last day (20th April 2021)")
plt.ylabel("Temperature (C)")
plt.show()
