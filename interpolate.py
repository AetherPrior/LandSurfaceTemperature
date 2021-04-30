import os
import numpy as np
import pandas as pd
from re import search


def interpolate(input_dir, output_dir):
    l = os.listdir(path=input_dir)
    for f in l:
        if search("csv", f):
            df = pd.read_csv(os.path.join(input_dir, f))
            df.columns = ['date', 'lat', 'long', 'temp']
            df['temp'] = df['temp'].replace(
                0, np.nan).interpolate().fillna(method="bfill")*0.02-273.16
            df.to_csv(os.path.join(output_dir, f), index=True,
                      columns=['lat', 'long', 'temp'])


if __name__ == '__main__':
    input_dir = input("Enter input dir of dataset: ")
    output_dir = input("Enter output dir for interpolated dataset: ")
    interpolate(input_dir, output_dir)
