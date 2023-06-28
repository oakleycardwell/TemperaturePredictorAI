###Temperature Prediction Using LSTM

This project demonstrates how to predict future temperature data using a Long Short-Term Memory (LSTM) network. The LSTM model is trained on historical weather data and is then used to forecast future temperatures.

#Project Overview

The project is split into two parts:

Training the LSTM model on historical data.

Testing the LSTM model on new unseen data and comparing the model's predictions with the actual temperatures.
The project utilizes Python libraries such as TensorFlow (for building the LSTM model), pandas (for data handling), scikit-learn (for data scaling), and matplotlib (for visualization).

#How It Works

The program reads historical weather data (date, time, and temperature), scales the temperature data using a MinMaxScaler to aid the neural network's learning process, and creates sequences of 10 consecutive hours of temperature data. The LSTM model uses each sequence to predict the temperature of the 11th hour.

After training the LSTM model, the program stores the trained model and the scaler for future use.

The testing part of the program involves loading the trained model and scaler, reading new weather data, and scaling the temperature data using the previously saved scaler. The program then prepares the data in the same sequence format used during training and feeds these sequences into the model to predict temperatures. The predicted temperatures are then rescaled to their original scale for comparison with the actual temperatures.

#Files

weather_data.csv: This is the training data file that contains historical weather data.
testing_weather_data.csv: This is the testing data file that contains new weather data for testing the model.
model_v1.h5: This is the trained LSTM model.
temp_scaler_v1.gz: This is the saved MinMaxScaler object used to scale the temperature data.
Getting Started
You can clone this repository and run the scripts Data_Trainer.py and Data_Tester.py on your local machine.

#Requirements

You need to have Python (version 3.6 or later) installed on your system, along with the following Python libraries:

TensorFlow
Keras
pandas
scikit-learn
matplotlib
joblib
You can install these packages using pip:

pip install tensorflow keras pandas scikit-learn matplotlib joblib

#Running the Scripts

Run Data_Trainer.py to train the LSTM model.
Run Data_Tester.py to test the LSTM model on new data.
Note: Ensure that the required data files (weather_data.csv and testing_weather_data.csv) are in the correct location as specified in the scripts.

#Output

The scripts plot the following graphs:

Training and validation loss curves during the model's training process.
A comparison of the original and predicted temperatures for the testing data.
These plots help visualize the model's learning process and its prediction performance.

#Contributing

Contributions are welcome. Please open an issue to discuss your suggestions or make a pull request with your improvements.

#License

This project is open source and available.
