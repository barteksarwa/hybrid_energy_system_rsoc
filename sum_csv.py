import os
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment

# Input directory for CSV files
csv_directory = 'C:\\Users\\Lenovo\\Documents\\python_projects\\thesis\\project\\simulation\\doe_output_csv_update_fff_11'

# Output CSV file path for sums
output_sums_csv_path = 'C:\\Users\\Lenovo\\Documents\\python_projects\\thesis\\project\\simulation\\sums_output_update_11.csv'

# Output Excel file path for sums
output_sums_excel_path = 'C:\\Users\\Lenovo\\Documents\\python_projects\\thesis\\project\\simulation\\sums_output_update_11.xlsx'

# Path to the 'denormalized_designs' file
designs_denormalized_file_path = 'C:\\Users\\Lenovo\\Documents\\python_projects\\thesis\\project\\simulation\\denormalized_designs_fff.xlsx'

# Path to the 'designs' file
designs_file_path = 'C:\\Users\\Lenovo\\Documents\\python_projects\\thesis\\project\\simulation\\designs_fff.xlsx'

# Optimization file 'designs_with_losses' path
optimization_file_path = 'C:\\Users\\Lenovo\\Documents\\python_projects\\thesis\\project\\simulation\\designs_with_losses_fff.xlsx'

# List all CSV files in the input directory with the name pattern 'pv_plot'
csv_files = sorted([f for f in os.listdir(csv_directory) if f.startswith('pv_plot') and f.endswith('.csv')])

# Check if there are any CSV files in the specified directory
if not csv_files:
    print("No CSV files found in the directory.")
else:
    # Initialize an empty list to store DataFrames
    dfs = []

    # Get the column names from the first CSV file
    first_file_path = os.path.join(csv_directory, csv_files[0])
    first_df = pd.read_csv(first_file_path, delimiter=' ')
    column_names = first_df.columns.tolist()

    # Loop through each CSV file and sum all columns
    for csv_file in csv_files:
        file_path = os.path.join(csv_directory, csv_file)

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path, delimiter=' ')

        # Sum all columns and append the sums to the list
        dfs.append(df.sum())

    # Concatenate the list of DataFrames into a single DataFrame
    sums_df = pd.concat(dfs, axis=1).T.reset_index(drop=True)

    # Save the sums DataFrame to an Excel file with column names and text wrapping for the first column
    with pd.ExcelWriter(output_sums_excel_path, engine='openpyxl') as writer:
        sums_df.to_excel(writer, index=False, header=True, sheet_name='Sheet1')
        
        # Access the workbook and the active sheet
        workbook = writer.book
        sheet = workbook['Sheet1']

        # Apply text wrapping to the first column and set row height
        for row in sheet.iter_rows(min_row=1, max_row=1, min_col=1, max_col=1):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True)
                sheet.row_dimensions[cell.row].height = 60

    print(f"Sums of all columns saved to '{output_sums_csv_path}' (CSV) and '{output_sums_excel_path}' (Excel).")

    # Load the 'designs' file with the correct encoding
    designs_df = pd.read_excel(designs_file_path)
    

    # Add sums of the last two columns to the 'designs' file separately
    designs_df['Energy deficit (kWh)'] = sums_df.iloc[:, -3] / 1000
    designs_df['Energy loss (kWh)'] = sums_df.iloc[:, -2] / 1000

    
    denom_designs_df = pd.read_excel(designs_denormalized_file_path)
    costs = [660, 2000, 1200, 5000]
    
    designs_df['Sum_Products_Cost'] = np.sum(denom_designs_df.iloc[:, :4].values * costs, axis=1)

    # Save the modified DataFrame into a new Excel file
    designs_df.to_excel(optimization_file_path, index=False)

    print(f'DataFrame with losses saved to {optimization_file_path}')