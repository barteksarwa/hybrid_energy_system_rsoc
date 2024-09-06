import os
import numpy as np
import pandas as pd


# Path to the file with designs
designs_denormalized_file_path = 'designs_testing_denormalized.xlsx'

# Path to the file with normalized designs
designs_file_path = 'designs_testing.xlsx'

# Directory with results of simulation for each design
csv_directory = 'doe_output_csv_test_loop'

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
        df = df[:-1]

        # Read the CSV file into a DataFrame
        sum_positive_indexes = np.sum(np.where(df.iloc[:, -3] > 0, 1, 0))
        sum_positive_indexes_list.append(sum_positive_indexes)
        df_np = df.to_numpy()
        s_row = np.sum(df_np, axis=0)

        # Sum all columns and append the sums to the list
        dfs.append(s_row)

    dfs = np.array(dfs)
    dfs = dfs[:, -3:]/1000

    # Recalculate energy loss and deficit into hydrogen
    energy_deficit = dfs[:, 0:1]
    energy_content_hydrogen = 33.3 # kWh / kg
    efficiency = 0.6
    hydrogen_amount_kg = energy_deficit / (energy_content_hydrogen * efficiency)
    dfs = np.concatenate((dfs, hydrogen_amount_kg), axis=1)

    energy_loss = dfs[:, 1:2]
    hydrogen_amount_kg_loss = energy_loss / (energy_content_hydrogen * efficiency)
    dfs = np.concatenate((dfs, hydrogen_amount_kg_loss), axis=1)
    dfs = np.concatenate((designs_normalised, dfs),axis=1)
    denom_designs_df = pd.read_excel(designs_denormalized_file_path)


    # Costs of the system
    costsunit_i = np.array([165, 300, 660, 1300])
    costs_r = np.array([0, 4, 8, 0])
    costs_install = np.array([1000, 200 * costs_r[1], 500 * costs_r[2], 200])
    costs_m = np.array([100, 100, 200, 0])
    price_hydrogen_kg = 6.40*4*3

    cost_capex = np.sum(denom_designs_df.iloc[:, :4].values * costsunit_i +
                        denom_designs_df.iloc[:, :4].values * (costs_r * costsunit_i) 
                        + costs_install, axis=1)[:, np.newaxis]
    total_costs = cost_capex + np.sum(costs_m * 30)
    dfs = np.concatenate((dfs, total_costs), axis=1)


    # Objective functions
    F2 = total_costs # * PRi / 8760
    F3 = (hydrogen_amount_kg*30*price_hydrogen_kg) + (hydrogen_amount_kg_loss*30*price_hydrogen_kg*0.2)
    LPSR = np.array([value / 8760 for value in sum_positive_indexes_list]).reshape(-1, 1)
    F1 = total_costs + (hydrogen_amount_kg*30*price_hydrogen_kg)  \
        + (hydrogen_amount_kg_loss*30*price_hydrogen_kg*0.4)
    dfs = np.concatenate((dfs, F1, F2, F3, LPSR), axis=1)

    np.savetxt("doe_test.csv", dfs, delimiter=" ", header="PV Battery SOFC TANK energy_deficit "
                                                          "Energy_loss System_cost F2 F3 LPSR F1")
