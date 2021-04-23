from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 

def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.80)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    time = np.linspace(1,len(test),len(test))
    plt.plot(time,predictions)
    plt.plot(time,test)
    plt.show()
    plt.clf()
    error = mean_squared_error(test, predictions)
    return error

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                mse = evaluate_arima_model(dataset, order)
                if mse < best_score:
                    best_score, best_cfg = mse, order
                print('ARIMA%s MSE=%.3f' % (order,mse))
                #continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))

filename = input("Enter file: ")
df = pd.read_csv(filename)
df.columns = ['date','lat','long','temp']
X = df['temp'].to_numpy()
p = [3]
d = [1,0]
q = [3]
evaluate_models(X,p,d,q)

# ARIMA 1,0,2 or 2,0,2