from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os


def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.90)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    if os.path.isfile("ARIMA.pkl"):
        print("LOADING")
        model_fit = ARIMAResults.load("ARIMA.pkl")
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        if not os.path.isfile("ARIMA.pkl"):
            model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    time = np.linspace(1, len(test), len(test))
    plt.xlabel("Days")
    plt.ylabel("Temperature (C)")
    line1, = plt.plot(time, predictions)
    line2, = plt.plot(time, test)
    plt.legend([line1, line2], ["ARIMA prediction", "Target"])
    plt.savefig("plots/ARIMA_fit.png")
    plt.clf()
    error = mean_squared_error(test, predictions)
    model_fit.save('ARIMA.pkl')
    return model_fit, error, predictions

# evaluate combinations of p, d and q values for an ARIMA model


def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                model_fit, mse, pred = evaluate_arima_model(dataset, order)
                if mse < best_score:
                    best_score, best_cfg = mse, order
                print('ARIMA%s MSE=%.3f' % (order, mse))
                # continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))


filename = input("Enter file: ")
df = pd.read_csv(filename)
df.columns = ['date', 'lat', 'long', 'temp']
X = df['temp'].to_numpy()
p = 1
d = 0
q = 2
model_fit, error, pred = evaluate_arima_model(X, (p, d, q))
pred_x = np.linspace(1, 10, 10)

pred_y = model_fit.forecast(steps=10)
print(pred_y)
plt.clf()
plt.plot(pred_x, pred_y)
plt.xlabel("Predictions since the last day (20th April 2021)")
plt.ylabel("Temperature (C)")
plt.savefig("plots/ARIMA_forecast_10_days.png")
plt.show()


# ARIMA 1,0,2 or 2,0,2
