# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 14:07:51 2023

@author: Lenovo
"""

from pyDOE2 import fullfact
import pandas as pd

EXCEL_FILE_NAME = 'designs_testing.xlsx'

# Define the levels for each factor
factor_levels = {
    "Photovoltaic Modules (number)": [i for i in range(16, 61)],
    "Battery Power (number)": [i for i in range(2, 15)],
    "Solid Oxide Stack (number of cells)": [i for i in range(6, 45)],
    "Hydrogen Storage Tanks (number of tanks)": [i for i in range(8,51)]
}

# Create a list of factors and their corresponding levels
factors = list(factor_levels.keys())
levels = [factor_levels[f] for f in factors]

# Generate a full factorial design
design = fullfact([3,3,3,3])-1

# Convert the design matrix to a DataFrame
df = pd.DataFrame(design, columns=factors)

# Print the design matrix
df.to_excel(EXCEL_FILE_NAME, index=False)

for factor in factors:
    lower_bound = min(factor_levels[factor])
    upper_bound = max(factor_levels[factor])
    scaling_factor = (upper_bound - lower_bound) / 2.0  # Scale factor is half of the range

    df[factor] = df[factor].apply(lambda x: x * scaling_factor + (lower_bound + upper_bound) / 2)

# Save the denormalized design matrix to an Excel file
denormalized_excel_file_name = f'{EXCEL_FILE_NAME.split(".")[0]}_denormalized.xlsx'
df.to_excel(denormalized_excel_file_name, index=False)
