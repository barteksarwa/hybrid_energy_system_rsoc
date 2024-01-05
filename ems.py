# -*- coding: utf-8 -*-
"""
2023

@author: Bartlomiej Sarwa
"""

from devices.global_constants import SOC_min, SOC_max

def ems(p_l, p_pv, SOC, p_bess, hydrogen_SOC, previous_s):
    p_l = -abs(p_l)
    s = previous_s   
    if p_l+p_pv >= 0:
        if SOC < SOC_max:
            s = 1 # BESS charges
        elif SOC == 0.8 and hydrogen_SOC < 1.0: 
            s = 2
        else:
            s = 7
    else:
        if SOC > SOC_min and p_pv + p_bess + p_l >= 0:
            s = 3
        elif SOC > SOC_min and p_pv + p_bess + p_l < 0:
            s = 5
        elif SOC == SOC_min and hydrogen_SOC == 0:
            s = 6
        elif SOC == SOC_min and hydrogen_SOC > 0:
            s = 4
            
    if SOC >= SOC_max and s != 2:
        SOC = SOC_max
    return s

# print(ems(1000, 1500, 0.8, 1500, 0.8, 1))

# s = 1 - P_pv > P_l, SOC < SOC_max - charge battery
# s = 2 - P_pv > P_l, SOC == SOC_max - produce hydrogen in SOEC
# s = 3 - P_pv < P_l, SOC > SOC_min, P_bess+P_pv > P_load  - draw power from battery
# s = 4 - P_pv < P_l, SOC == SOC_min, P_bess+P_pv < P_load - generate power in fuel cell
# s = 5 - P_pv < P_l, SOC > SOC_min, P_bess+P_pv < P_load - draw power from battery and generate power in fuel cell
# s = 6 - SOC == SOC_min and hydrogen_SOC == hydrogen_SOC_min - deficit of energy
# s = 7 - SOC == SOC_max and hydrogen_SOC == hydrogen_SOC_max - loss of energy