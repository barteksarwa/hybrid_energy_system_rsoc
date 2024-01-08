from .global_constants import *


class Battery:
    def __init__(self):
        pass

    def bess_discharge_capacity(self, qb, bd, ub, g, g_min, t):
        pb = qb * bd
        discharge_capacity = min(pb * t, ub * (g - g_min))
        return discharge_capacity

    def bess_charge_capacity(self, qb, bc, ub, g, g_max, t):
        pb = -qb * bc
        charge_capacity = max(pb * t, -ub * (g_max - g))
        return charge_capacity


# Test
# discharge_capacity = bess_discharge_capacity(qb, bd, ub, g, g_min, d)
# charge_capacity = bess_charge_capacity(qb, bc, ub, g, g_max, d)

# print(bess_discharge_capacity(qb, bd, ub, g, g_min, d))
# print(charge_capacity, discharge_capacity)
# print(bess_discharge_capacity(500.0, 0.97, 50000.0, 0.3, 0.2, 1.0))