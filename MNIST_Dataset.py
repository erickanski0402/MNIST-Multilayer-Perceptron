import csv
import numpy as np

def load_mnist():
    return np.loadtxt(open('C:/Users/erick/Desktop/train.csv', 'rb'), delimiter = ',', skiprows = 1)

def get_targets(rawData):
    return rawData[:,0]

def get_data(rawData):
    return rawData[:,1:]
