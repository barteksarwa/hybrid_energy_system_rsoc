from .global_constants import *


class Battery:
    def __init__(self):
        pass

    def bess_discharge_capacity(self, qb_in, bd_in, ub_in, g_in, g_min_in, d_in):
        pb = qb_in * bd_in
        discharge_capacity = min(pb * d_in, ub_in * (g_in - g_min_in))
        return discharge_capacity

    def bess_charge_capacity(self, qb_out, bc_out, ub_out, g_out, g_mean_out, d_out):
        pb_out = -qb_out * bc_out
        charge_capacity = max(pb_out * d_out, -ub_out * (g_mean_out - g_out))
        return charge_capacity


# Test
# discharge_capacity = bess_discharge_capacity(qb, bd, ub, g, g_min, d)
# charge_capacity = bess_charge_capacity(qb, bc, ub, g, g_max, d)

# print(bess_discharge_capacity(qb, bd, ub, g, g_min, d))
# print(charge_capacity, discharge_capacity)
# print(bess_discharge_capacity(500.0, 0.97, 50000.0, 0.3, 0.2, 1.0))