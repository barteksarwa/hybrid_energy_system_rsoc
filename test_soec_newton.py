# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 14:02:57 2023

@author: Lenovo
"""
from devices.global_constants import *
import devices.sofc as sofc
import numpy as np
import matplotlib.pyplot as plt

sofc_dev = sofc.SolidOxideFuelCell()


net_power = -400
soch2 = 1
n_cells = 5000

p_fc = min(25 * n_cells * a_cell, abs(net_power))
# if net_power[i] > 100:
j0 = 10 # punkt przed p_max
j = sofc_dev.newton_method(sofc_dev.w_sofc_diff, 1100, j0, 115000, p_fc, 100)
consh2 = np.minimum(sofc_dev.hydrogen_consumption_rate(j)
    * 22.4 * 3600, soch2 * capacityh2) 
dsoc = 0 / ub
dsoch2 = -consh2 / capacityh2
print()
print(consh2)
print(p_fc)
print(j)


# Calculate power-voltage characteristics
voltages = np.linspace(0, 1.2, 100)  # Adjust the range based on your system
powers = []

for voltage in voltages:

    j = sofc_dev.newton_method(sofc_dev.w_sofc_diff, 1100, j0, 115000, p_fc, voltage)
    power = voltage * j * n_cells * sofc.a_cell
    powers.append(power)

# Plot the characteristics
plt.plot(voltages, powers, label='Power-Voltage Characteristics')
plt.xlabel('Voltage (V)')
plt.ylabel('Power (W)')
plt.title('SOFC Power-Voltage Characteristics')
plt.legend()
plt.grid(True)
plt.show()
