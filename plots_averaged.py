import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
import InitParams


time = pd.read_excel('Load.xlsx', index_col=0, usecols='A')
time = time.index.to_numpy()

(li_ion_capacity, power, sofc_power, soec_power, battery_power, load, SoCH2, EMS_State,
 net_power, deficit_energy, loss_energy) = np.loadtxt("result_article_before\\before.csv", unpack=True)
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
#
data.set_index('time', inplace=True)
daily_data = data.resample('D').mean()

plt.figure()
plt.plot(daily_data.index, daily_data['soc'], label='$SoC_{\mathrm{BESS}} [-]$')
plt.plot(daily_data.index, daily_data['SoCH2'], label='$SoC_{\mathrm{H_{2}}} [-]$')

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

# plt.xlabel('Time')
plt.ylabel('Daily averageg $SoC$ [-]')
# plt.title('Daily averaged state of charge over the year for --- optimization by the F4 function')
plt.legend(loc=8,bbox_to_anchor=(0.5, -0.32), fancybox=False, shadow=False, ncol=2)
plt.grid(False)
plt.setp(ax.get_xticklabels(minor=True), rotation=45, ha='right')

plt.tight_layout()
# plt.show()
plt.savefig('soc_averaged_before.pdf', format='pdf')

