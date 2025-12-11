import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1. Data loading and merging
price_data = pd.read_csv('nodal price.csv', parse_dates=['timestamp'], index_col=0)
weather_data = pd.read_csv('weather.csv', parse_dates=['timestamp'], index_col=0)
merged_data = pd.merge(price_data, weather_data, on='timestamp')

# 2. Feature engineering (example feature list)
features = ['node_price', 'wind_speed', 'solar_irradiance']
target = 'node_price'  # Prediction target: next time-step price

# 3. Normalization (eliminate scale differences)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(merged_data[features])

# 4. Build time-series samples (time window = 24 hours)
def create_dataset(data, n_steps=24):
    X, y = [], []
    for i in range(len(data) - n_steps - 1):
        X.append(data[i:(i + n_steps), :])  # Include all features
        y.append(data[i + n_steps, 0])      # Target is the price column
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data)
print(f"Sample shape: {X.shape}")  # Output (num_samples, 24, 3)

# 5. Split training and test sets (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# 1. Model architecture design
model = Sequential([
    # First LSTM layer (return full sequence for the next layer)
    LSTM(
        64,
        activation='tanh',
        input_shape=(X_train.shape[1], X_train.shape[2]),
        return_sequences=True
    ),
    Dropout(0.2),  # Prevent overfitting

    # Second LSTM layer (return only final output)
    LSTM(32, activation='tanh'),
    Dropout(0.2),

    # Output layer (linear activation for regression)
    Dense(1)
])

# 3. Model compilation configuration
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# 4. Early-stopping callback settings
early_stop = EarlyStopping(monitor='val_loss', patience=5)

# 1. Model training (validation split = 20%)
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# 2. Model evaluation (test set performance)
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {test_mae:.4f}")

# 3. Generate predictions
y_pred = model.predict(X_test)

# 4. Inverse-scaling to original data
y_test_actual = scaler.inverse_transform(
    np.concatenate(
        [y_test.reshape(-1, 1),
         np.zeros((len(y_test), 2))],
        axis=1
    )
)[:, 0]

y_pred_actual = scaler.inverse_transform(
    np.concatenate(
        [y_pred.reshape(-1, 1),
         np.zeros((len(y_pred), 2))],
        axis=1
    )
)[:, 0]

# 5. Result visualization (example)
import matplotlib.pyplot as plt
import matplotlib

# Set global font
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 18

plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label='Actual Price')
plt.plot(y_pred_actual, label='Predicted Price')
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.legend()
plt.show()
