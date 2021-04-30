import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras import Sequential
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from osgeo import gdal
import os

tiff = gdal.Open(
    "Dataset/LST_DATA/MOD11A1.006_LST_Night_1km_doy2021101_aid0001.tif")
rip = gdal.Translate("output.xyz", tiff, unscale=True)
gt = tiff.GetGeoTransform()
print(gt)
x_min = gt[0]
x_size = gt[1]
y_min = gt[3]
y_size = gt[5]
"""
mx, my = 50000, 60000  # coord
px = (mx - x_min)/x_size  # lat
py = (my - y_min)/y_size  # long
print(px, py)
"""
"""
filedir = input("Enter dir: ")
df = pd.read_csv(filename)
df.columns = ['date','lat','long','temp']
X = df['temp'].to_numpy()
"""
