# -*- coding: utf-8 -*-
"""
2023

@author: Bartlomiej Sarwa
"""

from devices.global_constants import SOC_min, SOC_max,ub
import math
def ems(p_l, p_pv, SOC, p_bess, hydrogen_SOC, previous_s):
    p_l = -abs(p_l)
    s = previous_s   
    net = p_l + p_pv 
    if net >= 0:
        if SOC < SOC_max and p_bess > net and (SOC_max-SOC)*ub<=net:
            s = 1 # BESS charges
        elif math.isclose(SOC, SOC_max, abs_tol=1e-6) and hydrogen_SOC < 1.0:
            s = 2
        elif math.isclose(SOC, SOC_max, abs_tol=1e-6) and math.isclose(hydrogen_SOC, 1.0, abs_tol=1e-6):
            s = 7
        elif SOC < SOC_max and p_bess < net and hydrogen_SOC < 1.0 and (SOC_max-SOC)*ub<=net:
            s = 8
        elif SOC < SOC_max and p_bess < net and hydrogen_SOC < 1.0 and (SOC_max-SOC)*ub<=net:
            s = 8
    else:
        if SOC > SOC_min and p_bess >= abs(net) and (SOC-SOC_min)*ub>=net:
            s = 3
        elif SOC > SOC_min and p_bess < abs(net) and hydrogen_SOC > 0:
            s = 5
        elif SOC > SOC_min and p_bess < abs(net) and math.isclose(hydrogen_SOC, 0, abs_tol=1e-6) and (SOC-SOC_min)*ub>=net:
            s = 3
        elif math.isclose(SOC, SOC_min, abs_tol=1e-6) and math.isclose(hydrogen_SOC, 0, abs_tol=1e-6):
            s = 6
        elif math.isclose(SOC, SOC_min, abs_tol=1e-6) and hydrogen_SOC > 0:
            s = 4
            
    if SOC >= SOC_max and s != 2:
        SOC = SOC_max
    return s

# print(ems(1000, 1500, 0.8, 1500, 0.8, 1))

# s = 1 - P_pv > P_l, SOC < SOC_max, p_bess > net, (SOC_max-SOC)*ub<=net - charge battery
# s = 2 - P_pv > P_l, SOC == SOC_max, hydrogen_SOC < 1.0 - produce hydrogen in SOEC
# s = 3 - P_pv < P_l, SOC > SOC_min, P_bess+P_pv > P_load  - draw power from battery
# s = 4 - P_pv < P_l, SOC == SOC_min, P_bess+P_pv < P_load - generate power in fuel cell
# s = 5 - P_pv < P_l, SOC > SOC_min, P_bess+P_pv < P_load - draw power from battery and generate power in fuel cell
# s = 6 - SOC == SOC_min and hydrogen_SOC == hydrogen_SOC_min - deficit of energy
# s = 7 - SOC == SOC_max and hydrogen_SOC == hydrogen_SOC_max - loss of energy
# s = 8 - SOC < SOC_max and SOC_h2 < SOC_H2_max and p_bess < p_net - produce hydrogen and charge battery