import json
from prettytable import PrettyTable
import os
import csv

# Get the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the JSON file
json_file_path = os.path.join(current_directory, "Files", "testing_weather_data.json")

# Load the data from the JSON file
with open(json_file_path) as file:
    data = json.load(file)

# Extract the relevant information from the data
weather_records = data.get("days", [])

# Create a list to store the table rows
table_rows = []

# Create a table object
table = PrettyTable()
table.field_names = [
    "Date", "Time", "Temperature", "Humidity", "Dew",
    "Precipitation", "Precipitation Probability", "Wind Gust", "Wind Speed (mph)", "Wind Direction", "Pressure", "Visibility",
    "Cloud Cover", "Solar Radiation"
]

# Iterate over each record and add hourly updates to the table
for record in weather_records:
    date = record.get("datetime")
    hours = record.get("hours", [])

    # Iterate over hourly updates
    for hour in hours:
        time = hour.get("datetime")
        temp = hour.get("temp")
        humidity = hour.get("humidity")
        dew = hour.get("dew")
        precipitation = hour.get("precip")
        precipprob = hour.get("precipprob")
        windgust = hour.get("windgust")
        windspeed = hour.get("windspeed")
        winddir = hour.get("winddir")
        pressure = hour.get("pressure")
        visibility = hour.get("visibility")
        cloudcover = hour.get("cloudcover")
        solarradiation = hour.get("solarradiation")

        table_rows.append([
            date, time, temp, humidity, dew, precipitation, precipprob,
            windgust, windspeed, winddir, pressure, visibility,
            cloudcover, solarradiation
        ])

# Define the path to the output CSV file
csv_file_path = os.path.join(current_directory, "Files", "testing_weather_data.csv")

# Write the table data to the CSV file
with open(csv_file_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Date", "Time", "Temperature", "Humidity", "Dew", "Precipitation",
        "Wind Gust", "Wind Speed (mph)", "Wind Direction", "Pressure", "Visibility",
        "Cloud Cover", "Solar Radiation", "Conditions"
    ])
    writer.writerows(table_rows)

print("CSV file created successfully.")
