# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 14:07:51 2023

@author: Lenovo
"""

from pyDOE2 import fullfact
import pandas as pd

# Define the levels for each factor
factor_levels = {
    "Photovoltaic Modules (number)": [ i for i in range(6, 25)],
    "Battery Power (number)": [i for i in range(1, 12)],
    "Solid Oxide Stack (number of cells)": [ i for i in range(1, 16)],
    "Hydrogen Storage Tanks (number of tanks)": [ i for i in range(1,6)]
}

# Create a list of factors and their corresponding levels
factors = list(factor_levels.keys())
levels = [factor_levels[f] for f in factors]

# Generate a Plackett-Burman design
design = fullfact([3,3,3,3])-1

# Convert the design matrix to a DataFrame
df = pd.DataFrame(design, columns=factors)

# Print the design matrix
df.to_excel('designs_fff.xlsx', index=False)

for factor in factors:
    lower_bound = min(factor_levels[factor])
    upper_bound = max(factor_levels[factor])
    scaling_factor = (upper_bound - lower_bound) / 2.0  # Scale factor is half of the range

    df[factor] = df[factor].apply(lambda x: x * scaling_factor + (lower_bound + upper_bound) / 2)

# Save the denormalized design matrix to an Excel file
df.to_excel('denormalized_designs_fff_14.xlsx', index=False)

# 8, 4, (0.39223788+1)/2*31+5 ,(0.12311233+1)/2*5 +1