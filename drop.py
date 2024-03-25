# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 22:53:52 2024

@author: Lenovo
"""

import os
import pandas as pd

# Specify the directory containing the CSV files
directory_path = r'C:\Users\Lenovo\Documents\python_projects\thesis\project\simulation\doe_output_csv_update_fff'

# Iterate through each file in the directory
for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)

    # Check if the file is a CSV file
    if filename.endswith('.csv'):
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)

        # Drop the last row from the DataFrame
        df = df[:-1]

        # Save the modified DataFrame back to the CSV file
        df.to_csv(file_path, index=False)

# Print a message indicating the operation is completed
print("Last row deleted from each CSV file in the directory.")