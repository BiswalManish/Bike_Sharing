import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import warnings
warnings.filterwarnings('ignore')

#setting path
path = '/Users/manish/Documents/Projects/data_science/Bike_Sharing/data/raw data/day.csv'

day = pd.read_csv(path)

# print(day.head(10))

# print(day.info())

# print(day.describe())


# Creating cat and num cols
num_cols = ['temp', 'atemp', 'hum', 
            'casual', 'registered', 'cnt', 'windspeed']

cat_cols = [col for col in day.columns if col not in 
            num_cols and col != 'dteday' and col != 'instant']
