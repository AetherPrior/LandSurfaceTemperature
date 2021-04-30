import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm


def plot_series(filename):
    df = pd.read_csv(filename)
    df.columns = ['date', 'lat', 'long', 'temp']
    df['date'] -= df['date'][0]
    temp_diff = np.diff(df['temp'].to_numpy())
    # plot date time
    plt.plot(df['date'], df['temp'])
    plt.show()
    plt.clf()
    # plot first difference
    plt.plot(df['date'][:-1], temp_diff)
    plt.show()
    plt.clf()
    # plot histogram (appears stationary)
    plt.hist(df['temp'])
    plt.show()
    print(adfuller(df['temp']))
    plt.clf()
    # lag plots 1-7, we notice that the first lag has a relation but no further
    for i in range(1, 8):
        plt.subplot(2, 4, i)
        plt.scatter(df['temp'][0:-i], df['temp'][i:])
    plt.show()
    plt.clf()

    fig, ax = plt.subplots(2, 1)
    fig = sm.graphics.tsa.plot_acf(df['temp'], lags=400, ax=ax[0])
    ax[0].set_xlabel("Lags (Days)")
    ax[0].set_ylabel("Autocorrelation")
    fig = sm.graphics.tsa.plot_pacf(df['temp'], lags=400, ax=ax[1])
    ax[1].set_xlabel("Lags (Days)")
    ax[1].set_ylabel("Partial Autocorrelation")
    fig.tight_layout()
    plt.savefig("plots/ACF_PACF.png")
    plt.show()

    # We will probably need an ARIMA differnce term of 1, an AR of 2, and an MA of 1


if __name__ == '__main__':
    filename = input("Enter file: ")
    plot_series(filename)
