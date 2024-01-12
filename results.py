# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 00:55:57 2024

@author: Lenovo
"""

import numpy as np
import matplotlib.pyplot as plt

li_ion_capacity, power, sofc_power, soec_power, battery_power, load, SoCH2, EMS_State, net_power, deficit_energy, loss_energy = np.loadtxt("pv_plot_29/pv_plot_32.csv", unpack=True)
start = 0

stop = -1
# print(np.unique(EMS_State[start:stop]))
plt.plot(SoCH2[start:stop])
# defnegative = np.where(deficit_energy<-10)
print(np.sum(deficit_energy))
print(np.sum(loss_energy))
print(np.where(deficit_energy<0))
# print(load[155])
# print(sofc_power[defnegative])
# print(battery_power[defnegative])
# print(deficit_energy[defnegative])
# print(np.where(li_ion_capacity<0.2))
# print(np.where(EMS_State==0.0))
# plt.plot(li_ion_capacity[start:stop])
# indices = np.where((li_ion_capacity > 0.2) & (net_power < 0) & (SoCH2 > 0) & (battery_power < net_power))[0]
# print(f'Indices where conditions are met: {indices}')
print(f'EMS state at {start} is {EMS_State[start]}')
# print(f'SOC at {start} is {li_ion_capacity[start]}')
i = start
# plt.plot(battery_power[start:stop])
# print(load[i]-power[i]+soec_power[i]-sofc_power[i]-battery_power[i]-deficit_energy[i]+loss_energy[i])
# print(li_ion_capacity[i])
plt.savefig("out.pdf")