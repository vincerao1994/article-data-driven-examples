import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# 1. Data loading and preprocessing
# Assume all three data files are single-column CSV format
price = pd.read_csv('price.csv', header=None)
history_price = pd.read_csv('history price.csv', header=None)
wind = pd.read_csv('wind.csv', header=None)
solar = pd.read_csv('solar.csv', header=None)

# Merge data into a DataFrame
data = pd.DataFrame({
    'price': price[0].values,
    'wind': wind[0].values,
    'solar': solar[0].values,
    'history_price': price[0].values,
})

# Reshape to 3D array (days, hours, number of features)
data_3d = data.values.reshape((366, 24, 4))

# Split features (wind speed, irradiance, historical price) and target (price)
X = data_3d[:, :, 1:]  # shape (366, 24, 3)
y = data_3d[:, :, 0]   # shape (366, 24)

# Data normalization
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Feature normalization
X_flat = X.reshape(-1, 3)
X_scaled = scaler_X.fit_transform(X_flat)
X_scaled = X_scaled.reshape(366, 24, 3)

# Target normalization
y_flat = y.reshape(-1, 1)
y_scaled = scaler_y.fit_transform(y_flat)
y_scaled = y_scaled.reshape(366, 24, 1)

# 2. Build LSTM model
model = Sequential()
model.add(LSTM(96, activation='relu', return_sequences=True, input_shape=(24, 3)))
model.add(LSTM(32, activation='relu', return_sequences=True))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# 3. Train model
model.fit(X_scaled, y_scaled, epochs=100, batch_size=32, validation_split=0.1)

# 4. Typical-day prediction
def predict_typical_day(wind_solar_data):
    """Input typical-day weather data and return predicted prices."""
    # Preprocess input data
    input_data = np.array(wind_solar_data).reshape(1, 24, 3)
    input_scaled = scaler_X.transform(input_data.reshape(-1, 3)).reshape(1, 24, 3)

    # Make prediction
    predicted = model.predict(input_scaled)
    return scaler_y.inverse_transform(predicted.reshape(-1, 1)).flatten()

# Example usage: data format = 24 hours × [wind speed, irradiance, historical price]
typical_day_weather = [
    [2.64, 0, 331.5], [2.65, 0, 322.6], [2.69, 0, 315.3], [2.73, 0, 294.6], [2.78, 0, 278.7], [2.36, 0, 274.8],
    [2.59, 26, 294.3], [2.85, 94.6, 296], [3.15, 178.7, 310.9], [3.36, 258, 308.5],
    [3.41, 304.7, 302.2], [3.43, 329.9, 300],
    [3.44, 324.6, 251], [3.42, 295, 392.4], [3.38, 252.2, 323.6], [3.31, 223.9, 328],
    [3.04, 161.3, 343.7], [2.26, 108.1, 360.2],
    [2.29, 56.8, 367.3], [2.41, 0, 386.5], [2.49, 0, 356.6], [2.59, 0, 352.5], [2.69, 0, 341.4], [2.56, 0, 308.6],
]

# Run prediction
predicted_prices = predict_typical_day(typical_day_weather)

# 5. Save prediction results
output_df = pd.DataFrame({
    'Hour': range(1, 25),
    'Predicted_Price': predicted_prices
})
output_df.to_csv('predicted_prices.csv', index=False)
print("Prediction results have been saved to predicted_prices.csv")

# (Optional) Save the model for later use
model.save('price_prediction_model.h5')
