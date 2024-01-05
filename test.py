# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 14:02:57 2023

@author: Lenovo
"""
from devices.global_constants import *
import devices.sofc as sofc
import numpy as np

sofc_dev = sofc.SolidOxideFuelCell()
net_power = -400
soch2 = 1

p_fc = min(850 * n_cells_fc * a_cell_fc, abs(net_power))
# if net_power[i] > 100:
j0 = 1000 # punkt przed p_max
j = sofc_dev.newton_method(sofc_dev.w_sofc_diff, 1100, j0, 115000, p_fc)
consh2 = np.minimum(sofc_dev.hydrogen_consumption_rate(j)
    * 22.4 * 3600, soch2 * capacityh2) 
dsoc = 0 / ub
dsoch2 = -consh2 / capacityh2
print(dsoch2)
print(p_fc)
print(j)

