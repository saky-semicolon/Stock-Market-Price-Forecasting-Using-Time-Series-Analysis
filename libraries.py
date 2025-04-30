## 1. Importing Necessary Libraries and Dataset
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
import warnings

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

# Jupyter magic command (keep only if you're in a notebook)
# %matplotlib inline

# Ignore warnings
warnings.filterwarnings("ignore")

# Load the data
data = pd.read_csv("/content/drive/MyDrive/Datasets/Stock Price Pred/yahoo_stock.csv", parse_dates=['Date'])

print(data.info())

# Display the first few rows of the dataframe
data.head()

# Ensure the Date column is in datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Set the Date column as the index for the time series
data.set_index('Date', inplace=True)

# Ensure the data is sorted by date
data.sort_index(inplace=True)

# Decompose the Close column
ts_data_close = data['Close']

# Perform seasonal decomposition on the Close column
decomposition = sm.tsa.seasonal_decompose(ts_data_close, model='additive', period=365)
