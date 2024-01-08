import numpy as np
import matplotlib.pyplot as plt

li_ion_capacity, power, sofc_power, soec_power, battery_power, load, SoCH2, EMS_State, net_power, deficit_energy, loss_energy = np.loadtxt("test_plots_csv/test.csv", unpack=True)
start = 3209

stop = 3217
print(np.unique(EMS_State[start:stop]))
# plt.plot(SoCH2[start:stop])
print(np.where(li_ion_capacity<0.2))
print(np.where(EMS_State==1.0))
plt.plot(li_ion_capacity[start:stop])
# indices = np.where((li_ion_capacity > 0.2) & (net_power < 0) & (SoCH2 > 0) & (battery_power < net_power))[0]
# print(f'Indices where conditions are met: {indices}')
print(f'EMS state at {start} is {EMS_State[start]}')
# print(f'SOC at {start} is {li_ion_capacity[start]}')
i = start
# plt.plot(battery_power[start:stop])
# print(load[i]-power[i]+soec_power[i]-sofc_power[i]-battery_power[i]-deficit_energy[i]+loss_energy[i])
# print(li_ion_capacity[i])
plt.savefig("out.pdf")
