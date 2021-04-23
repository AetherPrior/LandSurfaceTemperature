import pmdarima as pmd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

filename = input("Enter file: ")
df = pd.read_csv(filename)
y = df['temp'].to_numpy()
y_train, y_test = y[0:int(len(y)*0.8)], y[int(len(y)*0.8):]
model = pmd.auto_arima(y_train,seasonal=True,m=12)
y_hat = model.predict(len(y)-int(len(y)*0.8))
time = np.linspace(1,len(y)-int(len(y)*0.8),len(y)-int(len(y)*0.8))
plt.plot(time,y_hat)
plt.plot(time,y_test)
plt.show()
