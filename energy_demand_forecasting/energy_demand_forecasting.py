import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Simulate energy demand data (hourly for 1 year)
np.random.seed(42)
dates = pd.date_range(start='2024-01-01', periods=8760, freq='H')
demand = np.random.normal(500, 50, 8760) + np.sin(np.arange(8760)/24) * 100  # Seasonal pattern
df = pd.DataFrame({'demand': demand}, index=dates)

# Preprocess data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['demand']])
X, y = [], []
sequence_length = 24
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i])
X, y = np.array(X), np.array(y)

# Split data
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(sequence_length, 1), return_sequences=True),
    LSTM(50, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

# Evaluate model
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)
print(f"Mean Absolute Error: {np.mean(np.abs(y_pred - y_test)):.2f} MW")

# Save model
model.save('energy_demand_model.h5')

# Plot results
plt.plot(y_test[:100], label='Actual')
plt.plot(y_pred[:100], label='Predicted')
plt.legend()
plt.savefig('demand_forecast.png')
