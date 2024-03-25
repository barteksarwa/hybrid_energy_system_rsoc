import os
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment

# Input directory for CSV files
csv_directory = 'doe_output_csv_update_fff_14'

# Output CSV file path for sums
output_sums_csv_path = 'sums_output_update_14.csv'

# Output Excel file path for sums
output_sums_excel_path = 'sums_output_update_14.xlsx'

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
    sum_positive_indexes_list = []
    # Loop through each CSV file and sum all columns
    for i in range(n_files):
        file_path = os.path.join(csv_directory, 'pv_plot_'+str(i)+'.csv')
        df = pd.read_csv(file_path, delimiter=' ')
        # Read the CSV file into a DataFrame
        sum_positive_indexes = np.sum(np.where(df.iloc[:, -2] > 0, 1, 0))
        
        sum_positive_indexes_list.append(sum_positive_indexes)
        df = np.loadtxt(file_path, delimiter=' ')
        s_row = np.sum(df, axis=0)

        # Sum all columns and append the sums to the list
        dfs.append(s_row)
    dfs = np.array(dfs)
    dfs = dfs[:, -2:]/1000

    energy_deficit = dfs[:, 0:1]
    # print(energy_deficit)
    # print(energy_deficit)
    conversion_factor = 3.6 # kWh to MJ
    # energy_deficit_mj = energy_deficit * conversion_factor
    energy_content_hydrogen = 33.3
    efficiency = 0.6
    hydrogen_amount_kg = energy_deficit / (energy_content_hydrogen * efficiency)
    dfs = np.concatenate((dfs, hydrogen_amount_kg), axis=1)
    # print(hydrogen_amount_kg)

    energy_loss = dfs[:, 1:2]
    # conversion_factor = 3.6 # kWh to MJ
    # energy_loss_mj = energy_loss * conversion_factor
    hydrogen_amount_kg_loss = energy_loss / (energy_content_hydrogen * efficiency)
    dfs = np.concatenate((dfs, hydrogen_amount_kg_loss), axis=1)
    # print(energy_loss)
    dfs = np.concatenate((designs_normalised, dfs),axis=1)

    # np.savetxt("DOE.csv", dfs, delimiter=" ", header="PV Battery SOFC TANK energy_deficit Energy_loss cost")
    denom_designs_df = pd.read_excel(designs_denormalized_file_path)
    costsunit_i = np.array([660, 1200, 1380, 5000])
    costs_r = np.array([0, 4, 8, 0])  # ilosc wymiany
    costs_install = np.array([2000, 300 * costs_r[1], 2000 * costs_r[2], 1000])
    # print(costs_install)
    costs_m = np.array([50, 100, 500, 250])
    price_hydrogen_kg = 6.40*4*3


    
    cost_capex = np.sum(denom_designs_df.iloc[:, :4].values * costsunit_i +
                        denom_designs_df.iloc[:, :4].values * (costs_r * costsunit_i) 
                        + costs_install, axis=1)[:, np.newaxis]
    # print(costs_install)
    total_costs = cost_capex + np.sum(costs_m * 30)
    # print(total_costs)
    dfs = np.concatenate((dfs, total_costs), axis=1)
    # print(total_costs)
    # print(dfs)
    F = total_costs + (hydrogen_amount_kg*30*price_hydrogen_kg+ \
        hydrogen_amount_kg_loss*30*price_hydrogen_kg*0.4)
    PRi = np.array([value / 8760 for value in sum_positive_indexes_list]).reshape(-1,1)
    F = F * (1-PRi/8760)
    
    # print(F)
    # print(dfs[0,:])
    # print(F[:,np.newaxis])
    dfs = np.concatenate((dfs, F), axis=1)

    # print(np.min(dfs[:,-1]))
    # print(dfs[0,-1])
    # print(dfs[-1, -1])
    # print(4200*0.5*30)
    np.savetxt("DOE14.csv", dfs, delimiter=" ", header="PV Battery SOFC TANK energy_deficit Energy_loss System_cost CF")
