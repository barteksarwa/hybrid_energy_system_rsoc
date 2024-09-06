"""
This script runs a simulation of a hybrid energy system in a loop. First it reads a table of designs generated using
Design of Experiments (DOE), gets the result for each design, and saves the results to CSV files.
The 'time' data is saved only for the first iteration.
"""

import numpy as np
import os
import pandas as pd
from simulation import f

# Create output directory
output_directory = 'doe_output_csv_test_loop'
os.makedirs(output_directory, exist_ok=True)


# Read possible designs of the system from the DOE for the simulation
df_designs = pd.read_excel('designs_testing_denormalized.xlsx')
design_table = df_designs.to_numpy()

for i, row in enumerate(design_table[75:], start=75):
    print(f'Simulating design no. {i}: ',*row)
    time_csv, output = f(*row)

    # Create file names with iteration number
    text_csv_filename = os.path.join(output_directory, f'pv_plot_{i}.csv')
    time_csv_filename = os.path.join(output_directory, f'time.csv')
    time_csv_str = np.datetime_as_string(time_csv)

    # Save files
    np.savetxt(text_csv_filename, output, header='li_ion_capacity PV_power SOFC_power SOEC_power battery_power load '
                                                 'SoCH2 EMS_State net_power energy_deficit energy_loss')
    if i == 1:  # Save 'time' only for the first iteration
        np.savetxt(time_csv_filename, time_csv_str, fmt='%s')

