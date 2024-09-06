import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
import InitParams


time = pd.read_excel('Load.xlsx', index_col=0, usecols='A')
time = time.index.to_numpy()

(li_ion_capacity, power, sofc_power, soec_power, battery_power, load, SoCH2, EMS_State,
 net_power, deficit_energy, loss_energy) = np.loadtxt("result_article_after\\2706_masterfinal.csv", unpack=True)
t = np.linspace(0,len(time), len(time))

data = pd.DataFrame({
    'time': time,
    'soc': li_ion_capacity,
    'power': power,
    'sofc_power': sofc_power,
    'soec_power': soec_power,
    'battery_power': battery_power,
    'load': load,
    'SoCH2': SoCH2,
    'EMS_State': EMS_State,
    'net_power': net_power,
    'deficit_energy': deficit_energy,
    'loss_energy': loss_energy
})

data['net'] = data['power'] + data['sofc_power'] - data['soec_power'] + data['battery_power'] - data['load']


data.set_index('time', inplace=True)
daily_data = data.resample('D').sum()

# Plot for avereged powers
plt.figure()
plt.plot(daily_data.index, daily_data['power'],color='gold', label='Daily generated PV energy [W]')
plt.plot(daily_data.index, daily_data['load'],color='darkorchid', label='Daily averaged load energy [W]')
plt.plot(daily_data.index, daily_data['sofc_power'], color='red',  label='Daily FC energy [W]')
plt.plot(daily_data.index, daily_data['soec_power'],color='lime', label='Daily EC energy [W]')
plt.plot(daily_data.index, daily_data['battery_power'], color='#000080', label='Daily battery energy [W]')
plt.plot(daily_data.index, daily_data['net'], color='pink', label='Daily net energy [W]', linewidth=1.5)

ax = plt.gca()
ax.xaxis.set_major_locator(MonthLocator())
ax.xaxis.set_major_formatter(DateFormatter(''))
# ax.xaxis.set_minor_locator(plt.NullLocator())

ticks = ax.get_xticks()
new_labels = []
new_ticks = []
for i in range(len(ticks) - 1):
    start_tick = ticks[i]
    end_tick = ticks[i + 1]
    mid_point = (start_tick + end_tick) / 2

    mid_date = daily_data.index[0] + pd.to_timedelta(mid_point, unit='D')
    label = mid_date.strftime('%b')

    new_ticks.append(start_tick)
    new_labels.append((mid_point, label))

# Set new tick positions at the beginning of the month
ax.set_xticks(new_ticks)
# Set new tick labels between the ticks
ax.set_xticks([pos for pos, label in new_labels], minor=True)
ax.set_xticklabels([label for pos, label in new_labels], minor=True)
ax.tick_params(axis='x', which='minor', length=0)
#

# plt.xlabel('Time')
plt.ylabel('Daily energy [Wh]')
legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=False, shadow=False, ncol=2)
plt.grid(False)
plt.tight_layout(pad=2.0)  # Adjust layout
plt.subplots_adjust(bottom=0.3)  # Add space at the bottom if necessary
plt.setp(ax.get_xticklabels(minor=True), rotation=45, ha='right')
plt.savefig('power_averaged_master.pdf', format='pdf', bbox_inches='tight', bbox_extra_artists=[legend])