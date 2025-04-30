"""## 3. Pre-processing"""

# Handle missing values (if any)
data.fillna(method='ffill', inplace=True)  # Forward fill

# Feature scaling (optional, but often beneficial)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
# Select numerical features to scale
numerical_features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Create lagged features (example: lag of 1 day for 'Close' price)
data['Close_Lag1'] = data['Close'].shift(1)
data.dropna(inplace=True) # Drop rows with NaN values created by shifting

# Optional: Feature engineering (e.g., rolling statistics)
data['Close_Rolling_Mean_7'] = data['Close'].rolling(window=7).mean()
data['Close_Rolling_Std_7'] = data['Close'].rolling(window=7).std()
data.dropna(inplace=True) # Drop rows with NaN values created by rolling mean/std

print(data.head())
print(data.info())
