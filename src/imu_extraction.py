import telemetry_parser
import pandas as pd
import numpy as np
from config import config
from pprint import pprint
from pathlib import Path

def extract_imu_data(file: Path):   	        # input is gopro video, sony, etc......
    tp = telemetry_parser.Parser(str(file))
    # Define the columns for the dataframe
    kolommen = ['TIMESTAMP', 'ACCL_x', 'ACCL_y', 'ACCL_z', 'GYRO_x', 'GYRO_y', 'GYRO_z']
    imu_data_df = pd.DataFrame(columns=kolommen)

    imu_data = tp.normalized_imu()  # Data retrieved from normalized_imu function, relevant data
    length = len(imu_data)  # Get the number of IMU data samples

    all_rows = []

    # Iterate over each telemetry sample (second)
    for i in range(length):
        timestamp_s = imu_data[i]['timestamp_ms'] / 1000  # Convert from ms to seconds
        
        # Retrieve accelerometer and gyroscope data for the current sample
        accl_data = imu_data[i]['accl']
        gyro_data = imu_data[i]['gyro']
        
        # Extract x, y, z components from the accelerometer and gyroscope data
        accl_x, accl_y, accl_z = accl_data
        gyro_x, gyro_y, gyro_z = gyro_data
        
        # Create a new row with the timestamp and sensor data
        new_row = [timestamp_s, accl_x, accl_y, accl_z, gyro_x, gyro_y, gyro_z]
        all_rows.append(new_row)

    imu_data_df = pd.DataFrame(all_rows, columns=kolommen)

    # Optionally save the DataFrame to a CSV file
    imu_data_df.to_csv(config.DATA_DIR / 'normalized_imu_data.csv', index=False)
    
    return imu_data_df