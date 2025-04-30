"""## 4. Train, Test and Evaluation"""

# Assuming 'data' is your DataFrame with 'Close' as a column and Date as the index
data.index = pd.to_datetime(data.index)

# Split the data
train_size = int(len(data) * 0.8)
test_start_dt = data.index[train_size]

# Create train and test sets
train = data[data.index < test_start_dt][['Close']]
test = data[data.index >= test_start_dt][['Close']]

print('Training data shape: ', train.shape)
print('Test data shape: ', test.shape)

!pip install pmdarima

# Scale data to be in range (0, 1)
scaler = MinMaxScaler()
train['Close'] = scaler.fit_transform(train[['Close']])
test['Close'] = scaler.transform(test[['Close']])

# Specify the number of steps to forecast ahead
HORIZON = 5  # Adjust as needed
print('Forecasting horizon:', HORIZON, 'days')

# Create a test data point for each HORIZON step
test_shifted = test.copy()
for t in range(1, HORIZON):
    test_shifted[f'Close+{t}'] = test_shifted['Close'].shift(-t)
test_shifted = test_shifted.dropna(how='any')

# Define the order and seasonal order for the SARIMAX model
order = (1, 1, 1)  # Perform AIC/BIC optimization to select the best parameters
seasonal_order = (1, 1, 1, 12)  # Adjust this if you have different seasonality

# Make predictions on the test data
training_window = 120  # Increased window size to capture more data
history = list(train['Close'])[-training_window:]

predictions = []
for t in range(len(test_shifted)):
    model = SARIMAX(history, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)

    yhat = model_fit.forecast(steps=HORIZON)
    predictions.append(yhat[0])

    obs = list(test_shifted.iloc[t])
    history.append(obs[0])
    history = history[-training_window:]

    if t % 100 == 0:  # Print progress every 100 steps
        print(f'Predicted {t+1}/{len(test_shifted)}')

# Inverse transform predictions and test data
predictions = np.array(predictions).reshape(-1, 1)
predictions = scaler.inverse_transform(predictions).flatten()
test_values = scaler.inverse_transform(test[['Close']])

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(test_values[:len(predictions)], predictions))
print(f'RMSE: {rmse}')

# Plot the results
plt.figure(figsize=(12,6))
plt.plot(train.index, scaler.inverse_transform(train[['Close']]), label='Train')
plt.plot(test.index[:len(predictions)], test_values[:len(predictions)], label='Test')
plt.plot(test.index[:len(predictions)], predictions, label='Predictions')
plt.legend()
plt.title('Stock Price Prediction (SARIMAX)')
plt.show()

# Evaluate the model using additional metrics
mae = mean_absolute_error(test_values[:len(predictions)], predictions)
print(f'MAE: {mae}')

# Calculate MAPE (Mean Absolute Percentage Error)
mape = np.mean(np.abs((test_values[:len(predictions)] - predictions) / test_values[:len(predictions)])) * 100
print(f'MAPE: {mape:.2f}%')


# Function to perform ADF test
def adf_test(timeseries):
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

# Test for stationarity on the training data
adf_test(train['Close'])

#Further analysis on residuals:
residuals = test_values[:len(predictions)] - predictions

# Plot the residuals
plt.figure(figsize=(10, 6))
plt.plot(residuals)
plt.title('Residuals of SARIMAX Model')
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.show()


# Histogram of residuals
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=30)
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

#Further analysis on residuals:
residuals = test_values[:len(predictions)] - predictions

# Plot the residuals
plt.figure(figsize=(10, 6))
plt.plot(residuals)
plt.title('Residuals of SARIMAX Model')
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.show()


# Histogram of residuals
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=30)
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# Model Diagnostics (Ljung-Box test for residual autocorrelation)
from statsmodels.stats.diagnostic import acorr_ljungbox

# Reshape residuals to 1D array
residuals = residuals.flatten()  # or residuals.reshape(-1)

lb_test = acorr_ljungbox(residuals, lags=[10, 20], return_df=True) # Adjust lags as needed
print(lb_test)


# Example: Check for heteroscedasticity using a rolling standard deviation
rolling_std = pd.Series(residuals).rolling(window=20).std() # Adjust window as needed
plt.plot(rolling_std)
plt.title('Rolling Standard Deviation of Residuals')
plt.show()

# Model Diagnostics (Ljung-Box test for residual autocorrelation - enhanced)
from statsmodels.stats.diagnostic import acorr_ljungbox

def check_residual_autocorrelation(residuals, max_lag=20):
    """Checks for autocorrelation in residuals using Ljung-Box test.

    Args:
        residuals (array-like): Time series of residuals.
        max_lag (int): Maximum lag for the Ljung-Box test.

    Returns:
        pandas.DataFrame: Results of the Ljung-Box test for multiple lags.
        Prints a summary of the results and potential issues.
    """

    lb_test = acorr_ljungbox(residuals, lags=range(1, max_lag + 1), return_df=True)
    print("Ljung-Box Test Results:\n", lb_test)

    # Identify lags with significant p-values (e.g., p < 0.05)
    significant_lags = lb_test[lb_test['lb_pvalue'] < 0.05].index.tolist()

    if significant_lags:
        print("\nSignificant autocorrelation detected at lags:", significant_lags)
        print("Consider adjusting model parameters (p, d, q, P, D, Q, s) or adding more features.")
    else:
        print("\nNo significant autocorrelation detected.")

    return lb_test

# Example usage:
residuals = test_values[:len(predictions)] - predictions
residuals = residuals.flatten()
lb_results = check_residual_autocorrelation(residuals)

def check_heteroscedasticity(residuals, window=20):
    """Checks for heteroscedasticity in residuals using a rolling standard deviation.

    Args:
      residuals (array-like): Time series of residuals.
      window (int): Window size for the rolling standard deviation.

    Returns:
      pandas.Series: Rolling standard deviation of residuals.
      Prints a summary of the results and potential issues.
    """

    rolling_std = pd.Series(residuals).rolling(window=window).std()
    plt.figure(figsize=(10, 6))
    plt.plot(rolling_std)
    plt.title('Rolling Standard Deviation of Residuals (Window = {})'.format(window))
    plt.xlabel("Time")
    plt.ylabel("Rolling Standard Deviation")
    plt.show()

    if rolling_std.std() > 0.1:  # Or another threshold based on data variability
        print("\nPossible heteroscedasticity detected. Residual variance changes over time.")
        print("Consider transforming the target variable or using a model that handles heteroscedasticity.")
    else:
        print("\nNo clear evidence of heteroscedasticity.")

    return rolling_std

# Example Usage
rolling_std = check_heteroscedasticity(residuals)

# Shift test predictions for plotting
testPredictPlot = np.empty_like(data['Close'].values.reshape(-1, 1))
testPredictPlot[:, :] = np.nan
testPredictPlot[look_back + len(train_data) : len(data['Close']) -1, :] = test_predict

# Prepare data for LSTM
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# Example usage:
look_back = 10  # Number of previous time steps to use for prediction
train_data = train['Close'].values.reshape(-1,1) # Reshape to 2D for scaling
test_data = test['Close'].values.reshape(-1, 1)

# Scale the data (if not already scaled)
scaler = MinMaxScaler(feature_range=(0, 1)) # Or use the same scaler from before
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

X_train, y_train = create_dataset(train_data, look_back)
X_test, y_test = create_dataset(test_data, look_back)

# Reshape input to be [samples, time steps, features] which is required for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1))) # You can experiment with different units and layers
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=2)  # Adjust epochs and batch_size

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Invert predictions back to original scale
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])


# Calculate root mean squared error
trainScore = np.sqrt(mean_squared_error(y_train[0], train_predict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = np.sqrt(mean_squared_error(y_test[0], test_predict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# Shift train predictions for plotting
trainPredictPlot = np.empty_like(train_data)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

# Shift test predictions for plotting
testPredictPlot = np.empty_like(data['Close'].values.reshape(-1, 1)) # Use data['Close'] shape
testPredictPlot[:, :] = np.nan
start_index = len(train_predict) + (look_back * 2) + 1
end_index = len(data['Close']) - 1  # Use len(data['Close'])
# Ensure end_index is greater than start_index
end_index = min(end_index, len(testPredictPlot) - 1)
# Check if the slice is valid
if start_index < end_index:
    testPredictPlot[start_index:end_index, :] = test_predict[:(end_index - start_index) +1 , :]
else:
    print("Warning: Test prediction slice is empty. Predictions will not be plotted.")

# Plot baseline and predictions
plt.plot(scaler.inverse_transform(train_data), label='Train Data')
plt.plot(trainPredictPlot, label='Train Predictions')
plt.plot(testPredictPlot, label='Test Predictions')
plt.legend()
plt.title('LSTM Stock Price Prediction')
plt.show()
