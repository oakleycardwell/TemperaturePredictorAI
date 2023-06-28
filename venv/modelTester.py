import numpy as np
import pandas as pd
import os
import joblib
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load your trained model
model = load_model('model_v1.h5')

# Get the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the CSV file
csv_file_path = os.path.join(current_directory, "Files", "testing_weather_data.csv")

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file_path, encoding="cp1252")

# Drop columns with nan values
df.dropna(axis=1, inplace=True)

# Preprocessing
# Parsing Date and Time into a single datetime column
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

# Set Datetime as index
df.set_index('Datetime', inplace=True)

# Drop Date and Time columns
df.drop(['Date', 'Time'], axis=1, inplace=True)

# Store original temperature data before scaling
original_temperatures = df['Temperature'].copy()

# Load the trained scaler
#scaler = joblib.load('scaler_v1.gz')
temp_scaler = joblib.load('temp_scaler_v1.gz')


# Scale 'Temperature'
df['Temperature'] = temp_scaler.transform(df[['Temperature']])

# Prepare data for the LSTM
lookback = 10
X = []
for i in range(lookback, len(df)):
    X.append(df.iloc[i-lookback:i, df.columns.get_loc('Temperature')]) # Only use 'Temperature' data for inputs
X = np.array(X)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape to be 3D for LSTM input

# Predict the temperature
predicted_temperature = model.predict(X)


# Since the output was scaled, we need to reverse the scale to get the actual temperature
predicted_temperature = temp_scaler.inverse_transform(predicted_temperature.reshape(-1, 1))

# Ensure original_temperatures has the same size as predicted_temperature
original_temperatures = original_temperatures[lookback:]

# Plot original and predicted temperatures
timestamps = df.index[lookback:]
plt.plot(timestamps, original_temperatures, label='Original Temperature')
plt.plot(timestamps, predicted_temperature.flatten(), label='Predicted Temperature')
plt.title('Original vs. Predicted Temperature')
plt.xlabel('Datetime')
plt.ylabel('Temperature')
plt.legend()
plt.show()