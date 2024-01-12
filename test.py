# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 14:04:22 2024

@author: Lenovo
"""
import pandas as pd

# Specify the path to your CSV file
csv_file_path = 'C:\\Users\\Lenovo\\Documents\\python_projects\\thesis\\project\\simulation\\doe_output_csv_update/pv_plot_1.csv'  # Replace with the actual path

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file_path, delimiter=' ')

# Print information about the DataFrame
print("DataFrame Info:")
print(df.info())

# Get the last two columns

# Sum the last two columns separately
sum_last_column = df.iloc[:, -2].sum()
sum_second_last_column = df.iloc[:, -3].sum()

# Display the results
print("Sum of the last column:", sum_last_column)
print("Sum of the second-to-last column:", sum_second_last_column)