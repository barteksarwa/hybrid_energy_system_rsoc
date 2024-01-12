import os
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment

# Input directory for CSV files
csv_directory = 'doe_output_csv_update_fff_11'

# Output CSV file path for sums
output_sums_csv_path = 'sums_output_update_11.csv'

# Output Excel file path for sums
output_sums_excel_path = 'sums_output_update_11.xlsx'

# Path to the 'denormalized_designs' file
designs_denormalized_file_path = 'denormalized_designs_fff.xlsx'

# Path to the 'designs' file
designs_file_path = 'designs_fff.xlsx'

# Optimization file 'designs_with_losses' path
optimization_file_path = 'designs_with_losses_fff.xlsx'

# List all CSV files in the input directory with the name pattern 'pv_plot'
csv_files = ([f for f in os.listdir(csv_directory) if f.startswith('pv_plot') and f.endswith('.csv')])
n_files = len(csv_files)

designs_normalised = pd.read_excel(designs_file_path)
designs_normalised = designs_normalised.to_numpy()

designs = pd.read_excel(designs_denormalized_file_path)
designs = designs.to_numpy()

# Check if there are any CSV files in the specified directory
if not csv_files:
    print("No CSV files found in the directory.")
else:
    # Initialize an empty list to store DataFrames
    dfs = []

    # Loop through each CSV file and sum all columns
    for i in range(n_files):
        file_path = os.path.join(csv_directory, 'pv_plot_'+str(i)+'.csv')

        # Read the CSV file into a DataFrame
        df = np.loadtxt(file_path, delimiter=' ')
        s_row = np.sum(df, axis=0)

        # Sum all columns and append the sums to the list
        dfs.append(s_row)
    dfs = np.array(dfs)
    dfs = dfs[:, -2:]/1000

    print(np.where(np.isnan(dfs[:,0])))
    energy_deficit = dfs[:, 0:1]
    dfs = np.concatenate((designs_normalised, dfs),axis=1)
    # np.savetxt("DOE.csv", dfs, delimiter=" ", header="PV Battery SOFC TANK energy_deficit Energy_loss cost")
    denom_designs_df = pd.read_excel(designs_denormalized_file_path)
    costs = [660, 2000*6, 1200*3, 5000]
    cost = np.sum(denom_designs_df.iloc[:, :4].values * costs, axis=1)[:,np.newaxis]
    dfs = np.concatenate((dfs, cost), axis=1)
    # print(dfs)
    F = cost + energy_deficit*30*0.5
    # print(dfs[0,:])
    # print(F[:,np.newaxis])
    dfs = np.concatenate((dfs, F), axis=1)
    print(np.min(dfs[:,-1]))
    # print(dfs[0,-1])
    # print(dfs[-1, -1])
    # print(4200*0.5*30)
    np.savetxt("DOE.csv", dfs, delimiter=" ", header="PV Battery SOFC TANK energy_deficit Energy_loss System_cost CF")
