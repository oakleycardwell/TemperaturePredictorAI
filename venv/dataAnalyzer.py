import os
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt


# Get the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the CSV file
csv_file_path = os.path.join(current_directory, "Files", "weather_data.csv")

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file_path, encoding="cp1252")

# Drop columns with nan values
df.dropna(axis=1, inplace=True)

# Preprocessing
# Parsing Date and Time into a single datetime column
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'].astype(str), format='%Y-%m-%d %H:%M:%S')


# Set Datetime as index
df.set_index('Datetime', inplace=True)

# Drop Date and Time columns
df.drop(['Date', 'Time'], axis=1, inplace=True)


# Scale the data
#scaler = MinMaxScaler(feature_range=(0, 1))
#scaled_data = scaler.fit_transform(df)

# Save temperature scaler
# Scale 'Temperature'
temp_scaler = MinMaxScaler(feature_range=(0, 1))
df['Temperature'] = temp_scaler.fit_transform(df[['Temperature']])

# Prepare data for the LSTM
lookback = 10
X, y = [], []
for i in range(lookback, len(df)):
    X.append(df.iloc[i-lookback:i, df.columns.get_loc('Temperature')]) # Only use 'Temperature' data for inputs
    y.append(df.iloc[i, df.columns.get_loc('Temperature')])  # Only use 'Temperature' data for outputs
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape to be 3D for LSTM input

# Split data into training and test set
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.3))
model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=100))
model.add(Dropout(0.3))
model.add(Dense(1))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model with validation split
history = model.fit(X_train, y_train, validation_split=0.2, epochs=30, batch_size=32)

#Display data
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Save the scaler object and trained model
#joblib.dump(scaler, 'scaler_v1.gz')
joblib.dump(temp_scaler, 'temp_scaler_v1.gz')
model.save('model_v1.h5')
