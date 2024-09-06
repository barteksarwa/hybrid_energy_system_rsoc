import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import sys
import os

# sys.path.append("..")
import InitParams

time = pd.read_excel('Load.xlsx', index_col=0, usecols='A')
time = time.index.to_numpy()
load = pd.read_excel('Load.xlsx', usecols='T')
load = pd.to_numeric(load.squeeze(), errors='coerce')

# FILES_DIRECTORY = r'C:\Users\Lenovo\Documents\python_projects\thesis\project\simulation\doe_output_csv'
# os.chdir(FILES_DIRECTORY)

# for csv_file in os.listdir(FILES_DIRECTORY):
#     if csv_file.endswith('.csv'):
#         csv_path = os.path.join(FILES_DIRECTORY, csv_file)
#         plots = np.loadtxt(f'{csv_file}')
#         lion_capacity = plots[:, 0]
#         power = plots[:, 1]
#         load = plots[:, 2]
#         soch2 = plots[:, 3]
#
#         range_ = (1400, 1510)
#         fig, ax1 = plt.subplots()
#
#         ax1.plot(time[range_[0]:range_[1]], power[range_[0]:range_[1]], 'c-',
#                   lw=0.75, label='PV panel production')
#         ax1.plot(time[range_[0]:range_[1]], load[range_[0]:range_[1]], 'm-',
#                   lw=0.75, label='Electric load profile')
#         ax1.set_xlabel('Time (date)')
#         ax1.set_ylabel('Energy (Wh)')
#         ax1.set_ylim([0, 2000])
#
#         ax2 = ax1.twinx()
#         ax2.plot(time[range_[0]:range_[1]], lion_capacity[range_[0]:range_[1]],
#                   'g-', lw=0.75, label='SOC battery')
#         ax2.plot(time[range_[0]:range_[1]], soch2[range_[0]:range_[1]],
#                   'r-', lw=0.75, label='SOC Hydrogen')
#         ax2.set_ylabel('SOC (-)')
#         ax2.set_ylim([-0.05, 1.05])
#
#         # Set the major locator and formatter for the x-axis
#         ax1.xaxis.set_major_locator(mdates.DayLocator())
#         ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#
#         handles1, labels1 = ax1.get_legend_handles_labels()
#         # handles2, labels2 = ax2.get_legend_handles_labels()
#
#
#         combined_handles = handles1 #+ handles2
#         combined_labels = labels1 # + labels2
#
#         # Show the legend under the plot
#         ax1.legend(combined_handles, combined_labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncols=2, fancybox=True, shadow=True)
#
#         plt.title('PV power production vs electric load profile', pad=15)
#
#         plt.show()
#

# Define the number of days in each month
days_in_month = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 30)

# Initialize a list to store the monthly load aggregates
monthly_load_aggregates = []

# Calculate the monthly aggregates
start_idx = 0
for days in days_in_month:
    # Slice the hourly load data for the current month
    month_data = load[start_idx:start_idx + (days * 24)]

    # Calculate the sum for the current month
    monthly_sum = np.sum(month_data)

    # Append the sum to the list of monthly load aggregates
    monthly_load_aggregates.append(monthly_sum)

    # Update the start index for the next month
    start_idx += days * 24

# The monthly_load_aggregates list now contains the aggregated values for each month
monthly_load_aggregates_tuple = tuple(monthly_load_aggregates)
print(monthly_load_aggregates_tuple)

# Months labels
months = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

# Formatter function to divide y-axis labels by 1000
def divide_by_1000(x, pos):
    return f'{int(x / 1000)}'

# Create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(months, monthly_load_aggregates_tuple, color='m')
plt.xlabel("Months")
plt.ylabel("Monthly Electricity Demand [kWh]")
plt.title("Generated Monthly Load Profile")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.gca().get_yaxis().set_major_formatter(FuncFormatter(divide_by_1000))
plt.xticks(rotation=45)  # Rotate month labels for better readability
plt.tight_layout()
plt.show()
