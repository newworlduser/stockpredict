import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Layer, Bidirectional, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from kerastuner.tuners import RandomSearch
import ta  # Technical analysis library

# 1. Data Acquisition and Feature Engineering
stock_symbol = 'ICICIBANK.NS'
data_range = '2y'
data = yf.download(stock_symbol, period=data_range)

# Add technical indicators using the ta library
data['RSI'] = ta.momentum.RSIIndicator(data['Close'].squeeze(), window=14).rsi()
data['MACD'] = ta.trend.MACD(data['Close'].squeeze()).macd()

data['BB_High'] = ta.volatility.BollingerBands(data['Close'].squeeze()).bollinger_hband()
data['BB_Low'] = ta.volatility.BollingerBands(data['Close'].squeeze()).bollinger_lband()

# Define the features to be used and drop missing values
feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'BB_High', 'BB_Low']
data = data[feature_columns].dropna()

# Scale the data to [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = scaler.fit_transform(data.values)

# 2. Create sequences from multivariate data
time_step = 60  # Using 60 days of data for each sample

def create_dataset_multivariate(dataset, time_step=1):
    X, Y = [], []
    close_index = feature_columns.index('Close')
    for i in range(len(dataset) - time_step):
        X.append(dataset[i:(i + time_step)])
        Y.append(dataset[i + time_step, close_index])
    return np.array(X), np.array(Y)

X, y = create_dataset_multivariate(dataset_scaled, time_step)

# 3. Time-based Train-Test Split (first 80% for training)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 4. Define a custom Attention Layer
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: (batch_size, time_steps, hidden_size)
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros", trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        output = tf.keras.backend.sum(output, axis=1)
        return output

# 5. Hyperparameter Tuning with Keras Tuner using RandomSearch
from tensorflow.keras.optimizers import Adam

def build_model(hp):
    model = Sequential()
    # First Bidirectional LSTM layer with L2 regularization, BatchNormalization and Dropout
    model.add(Bidirectional(LSTM(units=hp.Int('units_1', min_value=50, max_value=200, step=50),
                                 return_sequences=True,
                                 input_shape=(time_step, X_train.shape[2]),
                                 kernel_regularizer=l2(hp.Float('l2_reg', min_value=0.0001, max_value=0.01, sampling='log')))))
    model.add(BatchNormalization())
    model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)))

    # Second Bidirectional LSTM layer
    model.add(Bidirectional(LSTM(units=hp.Int('units_2', min_value=50, max_value=200, step=50),
                                 return_sequences=True,
                                 kernel_regularizer=l2(hp.Float('l2_reg', min_value=0.0001, max_value=0.01, sampling='log')))))
    model.add(BatchNormalization())
    model.add(Dropout(rate=hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)))

    model.add(AttentionLayer())
    model.add(Dense(units=hp.Int('dense_units', min_value=30, max_value=100, step=10), activation='relu'))
    model.add(Dense(1))  # Output layer for regression

    model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='mean_squared_error')
    return model

tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=2,
    directory='keras_tuner_dir',
    project_name='stock_price_prediction'
)

tuner.search(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1,
             callbacks=[EarlyStopping(patience=10)])
best_model = tuner.get_best_models(num_models=1)[0]

# 6. Train the Best Model with EarlyStopping and ModelCheckpoint
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
]
history = best_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1,
                         callbacks=callbacks, verbose=1)

# 7. Making Predictions on Training and Test Sets
train_predict = best_model.predict(X_train)
test_predict = best_model.predict(X_test)

# Helper function to inverse transform only the Close price
def inverse_transform(predictions, feature_index, scaler, n_features):
    dummy = np.zeros((len(predictions), n_features))
    dummy[:, feature_index] = predictions[:, 0]
    inv = scaler.inverse_transform(dummy)
    return inv[:, feature_index]

close_index = feature_columns.index('Close')
n_features = len(feature_columns)
train_predict_inv = inverse_transform(train_predict, close_index, scaler, n_features)
y_train_inv = inverse_transform(y_train.reshape(-1, 1), close_index, scaler, n_features)
test_predict_inv = inverse_transform(test_predict, close_index, scaler, n_features)
y_test_inv = inverse_transform(y_test.reshape(-1, 1), close_index, scaler, n_features)

# Evaluate with RMSE, MAE, and R-squared
train_rmse = np.sqrt(mean_squared_error(y_train_inv, train_predict_inv))
test_rmse = np.sqrt(mean_squared_error(y_test_inv, test_predict_inv))
test_mae = mean_absolute_error(y_test_inv, test_predict_inv)
test_r2 = r2_score(y_test_inv, test_predict_inv)

print(f"Train RMSE: {train_rmse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")
print(f"Test MAE: {test_mae:.2f}")
print(f"Test R-squared: {test_r2:.2f}")

# 8. Visualization of Actual vs. Predicted Prices
plt.figure(figsize=(12, 6))
plt.plot(data.index[time_step:train_size+time_step], y_train_inv, label='Train Actual')
plt.plot(data.index[time_step:train_size+time_step], train_predict_inv, label='Train Predicted')
plt.plot(data.index[train_size+time_step:], y_test_inv, label='Test Actual')
plt.plot(data.index[train_size+time_step:], test_predict_inv, label='Test Predicted')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title(f'{stock_symbol} Stock Price Prediction')
plt.legend()
plt.show()

# 9. Predicting the Next 10 Days' "Close" Prices
predicted_next = []
last_sequence = dataset_scaled[-time_step:]  # Use the last 60 days as the initial sequence

for i in range(10):  # Predict next 10 days
    x_input = np.expand_dims(last_sequence, axis=0)
    pred_scaled = best_model.predict(x_input)
    pred_unscaled = inverse_transform(pred_scaled, close_index, scaler, n_features)
    predicted_next.append(pred_unscaled[0])

    # Create a new day entry by updating the 'Close' value in the last day
    new_day = last_sequence[-1].copy()
    new_day[close_index] = pred_scaled[0, 0]
    # Update sequence: remove oldest day and append the new day
    last_sequence = np.vstack((last_sequence[1:], new_day.reshape(1, -1)))

print("Predicted prices for the next 10 days:")
for i, price in enumerate(predicted_next, start=1):
    print(f"Day {i}: {price:.2f}")

# Plot the predicted prices
plt.figure(figsize=(10, 6))
# Plot the last 30 actual days for context
last_30_days = data['Close'].values[-30:]
days_before = np.arange(-30, 0)
plt.plot(days_before, last_30_days, label='Last 30 Days (Actual)', color='blue')

# Plot the predicted 10 days
days_ahead = np.arange(len(predicted_next))
plt.plot(days_ahead, predicted_next, label='Next 10 Days (Predicted)', color='red', linestyle='--')

plt.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
plt.xlabel('Days (0 = Today)')
plt.ylabel('Stock Price')
plt.title(f'{stock_symbol} - 10 Day Price Prediction')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
