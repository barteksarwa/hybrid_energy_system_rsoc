# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 00:55:57 2024

@author: Lenovo
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
import pandas as pd
import InitParams
import calendar

def get_date_label(time_index, start):
    date = time_index[start]
    # Convert numpy.datetime64 to datetime object
    date = pd.Timestamp(date)
    month = calendar.month_name[date.month]
    day = date.day
    day_suffix = 'th' if 4 <= day <= 20 or 24 <= day <= 30 else ['st', 'nd', 'rd'][day % 10 - 1]
    return f"{month} {day}{day_suffix}"

# start = 2088+24 # March 28
# stop = 2112+24

time = pd.read_excel('Load.xlsx', index_col=0, usecols='A')
time = time.index.to_numpy()

li_ion_capacity, power, sofc_power, soec_power, battery_power, load, SoCH2, EMS_State, net_power, deficit_energy, loss_energy = np.loadtxt("result_article_before\\.csv", unpack=True)
t = np.linspace(0,len(time),len(time))

time_ranges = [
    (2112, 2136),       # March 28
    (4296, 4320),            # June
    (6504, 6528),            # September 28
    (7320, 7344)             # November 1
]

# 2x2 plots
fig = plt.figure()
gs = fig.add_gridspec(4, 2, height_ratios=[1, 3, 1, 3])

# Iterate through each time range and create the subplots
for idx, (start, stop) in enumerate(time_ranges):
    row = idx // 2
    col = idx % 2
    date_label = get_date_label(time, start)

    # Create upper subplot for state of charge data
    ax2 = fig.add_subplot(gs[row * 2, col])

    # Create lower subplot for power data
    ax1 = fig.add_subplot(gs[row * 2 + 1, col])

    # Plotting on the lower subplot (power data)
    ax1.plot(time[start:stop], power[start:stop], color='gold', label='Generated PV power [W]')
    ax1.plot(time[start:stop], load[start:stop], color='darkorchid', label='Load demand [W]')
    ax1.plot(time[start:stop], sofc_power[start:stop], color='red', label='SOFC power [W]')
    ax1.plot(time[start:stop], soec_power[start:stop], color='lime', label='SOEC power [W]')
    ax1.set_xlabel(date_label)
    ax1.set_ylabel('Power [W]', color='black')
    ax1.set_ylim([0, 6501])
    ax1.tick_params('y', colors='black')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(True)

    # Plotting on the upper subplot (state of charge data)
    ax2.plot(time[start:stop], SoCH2[start:stop], color='blue', label=r'$SOC_{\mathrm{H_{2}}} [-]$')
    ax2.plot(time[start:stop], li_ion_capacity[start:stop], color='black', label=r'$SOC_{\mathrm{BESS}} [-]$')
    ax2.set_ylabel('$SOC$ [-]', color='black')
    ax2.set_ylim([0, 1.05])
    ax2.tick_params('y', colors='black')
    ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(True)
    ax2.spines['right'].set_visible(True)

    # Set major locator to hours with a specified interval
    ax1.xaxis.set_major_locator(mdates.HourLocator(range(0, 24), 12))

    # Format x-axis with custom date format
    date_format = '%H:%M:%S'
    ax1.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    if col == 1:
        ax1.set_ylabel('')
        ax1.tick_params(labelleft=False)
        ax2.set_ylabel('')
        ax2.tick_params(labelleft=False)


# Combine legends
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
combined_handles = handles1 + handles2
combined_labels = labels1 + labels2

plt.legend(
    combined_handles, combined_labels, loc=9, bbox_to_anchor=(-0.2, -0.5),
    fancybox=False, shadow=False, ncol=2
)

# Set overall title
plt.suptitle('Simulation of the microgrid', y=0.98)

# Adjust layout to make room for the legend
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.subplots_adjust(hspace=0.38)

# Show the plot
# plt.show()
fig.savefig("combinedfourdays.pdf", format='pdf', bbox_inches='tight')

# # Create a figure and axis
# fig, ax1 = plt.subplots()
# cumulative_deficit_energy = np.cumsum(deficit_energy)
# # Plotting on the left y-axis
# ax1.plot(t[start:stop], li_ion_capacity[start:stop], label=r'$SOC_\mathrm{BESS}$ [-]', color='lime')
# # ax1.plot(t[start:stop], SoCH2[start:stop], label=r'$SOC_\mathrm{H_2}$ [-]', color='red')
# ax1.set_xlabel('Time [h]')
# ax1.set_ylabel('State of Charge [-]')
# ax1.tick_params('y')
#
# # Create a secondary y-axis
# ax2 = ax1.twinx()
# ax2.plot(t[start:stop], power[start:stop], label='PV power [W]',  color='Gold')
# ax2.plot(t[start:stop], load[start:stop], label='Load [W]',  color='purple')
# ax2.set_ylabel(' Energy [W]')
# ax2.tick_params('y')
# ax1.xaxis.set_major_locator(mdates.HourLocator(range(0, 24), 24))
#
# # Format x-axis with custom date format
# date_format = '%H:%M'
# ax1.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
#
# # Display legends
# handles1, labels1 = ax1.get_legend_handles_labels()
# handles2, labels2 = ax2.get_legend_handles_labels()
# combined_handles = handles1 + handles2
# combined_labels = labels1 + labels2
#
# plt.legend(
#     combined_handles, combined_labels, loc='upper center', bbox_to_anchor=(0.5, -0.15),
#     fancybox=True, shadow=True, ncol=3
# )


# Display the plot
# plt.show()

# Save the plot to a PDF file
# plt.savefig('soc.pdf', bbox_inches='tight')

# for i in range(start,stop):
#     print(np.where(SoCH2[i]!=SoCH2[i-1]))
#
# # Create a figure and axis
# fig = plt.figure(figsize=(10, 8))
#
# # Create a GridSpec with 2 rows and 1 column
# gs = fig.add_gridspec(2, 1, height_ratios=[1, 3], hspace=0.2)
#
# # Create the upper subplot (ax2)
# ax2 = fig.add_subplot(gs[0, 0])
#
# # Create the lower subplot (ax1)
# ax1 = fig.add_subplot(gs[1, 0])
#
# # Plotting on the left y-axis
# ax1.plot(time[start:stop], power[start:stop], color='gold', label='Generated PV power [W]')
# ax1.plot(time[start:stop], load[start:stop], color='darkorchid', label='Load demand [W]')
# ax1.plot(time[start:stop], sofc_power[start:stop], color='red', label='SOFC power [W]')
# ax1.plot(time[start:stop], soec_power[start:stop], color='lime', label='SOEC power [W]')
# ax1.set_xlabel('Time')
# ax1.set_ylabel('Power [W]', color='black')
# ax1.set_ylim([0, 6501])
# ax1.tick_params('y', colors='black')
# ax1.spines['top'].set_visible(False)
# ax1.spines['right'].set_visible(True)
# # ax1.yaxis.set_tick_params(right=True, labelright=False)
#
# # Create a secondary y-axis
#
# ax2.plot(time[start:stop], SoCH2[start:stop], color='blue', label=r'$SOC_{\mathrm{H_{2}}} [-]$')
# ax2.plot(time[start:stop], li_ion_capacity[start:stop], color='black', label=r'$SOC_{\mathrm{BESS}} [-]$')
# ax2.set_ylabel('State of Charge [-]', color='black')
# ax2.set_ylim([0, 1.05])
# ax2.tick_params('y', colors='black')
# ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
# ax2.set_xlim(ax1.get_xlim())
# ax2.spines['bottom'].set_visible(False)
# ax2.spines['top'].set_visible(True)
# ax2.spines['right'].set_visible(True)
# # ax2.yaxis.set_tick_params(right=True, labelright=False)
#
# # Set major locator to hours with a specified interval
# ax1.xaxis.set_major_locator(mdates.HourLocator(range(0, 24), 4))
#
# # Format x-axis with custom date format
# date_format = '%H:%M:%S'
# ax1.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
#
#
# # Set title
# plt.suptitle('Simulation of the microgrid November 1st')
#
# # Display legends
# handles1, labels1 = ax1.get_legend_handles_labels()
# handles2, labels2 = ax2.get_legend_handles_labels()
# combined_handles = handles1 + handles2
# combined_labels = labels1 + labels2
#
# plt.legend(
#     combined_handles, combined_labels, loc='upper center', bbox_to_anchor=(0.5, -0.08),
#     fancybox=False, shadow=False, ncol=3
# )
# # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
# #     fancybox=True, shadow=True, ncol=2
# # )
# fig.tight_layout(rect=[0, 0, 1, 0.95])
# # Show the plot
# plt.show()
# fig.savefig("rsofc23nov1.pdf", format='pdf', bbox_inches='tight')
#


# print(load[i]-power[i]+soec_power[i]-sofc_power[i]-battery_power[i]-deficit_energy[i]+loss_energy[i])
# print(li_ion_capacity[i])
