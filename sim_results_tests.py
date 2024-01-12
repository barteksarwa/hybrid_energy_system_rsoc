import numpy as np
import matplotlib.pyplot as plt

li_ion_capacity, power, sofc_power, soec_power, battery_power, load, SoCH2, EMS_State, net_power, deficit_energy, loss_energy = np.loadtxt("C:/Users/Lenovo/Documents/python_projects/thesis/project/simulation/doe_output_csv_update/pv_plot_5.csv", unpack=True)
start = 0
stop = -1
print(f'Unique states of EMS are {np.unique(EMS_State[start:stop])}')

for i in range(1, 43):
    file_path = f"C:/Users/Lenovo/Documents/python_projects/thesis/project/simulation/doe_output_csv_update/pv_plot_{i}.csv"

    # Load data from the current file
    li_ion_capacity, power, sofc_power, soec_power, battery_power, load, SoCH2, EMS_State, net_power, deficit_energy, loss_energy = np.loadtxt(file_path, unpack=True)

    # Print unique values of EMS_State
    unique_states = np.unique(EMS_State)
    print(f"Unique EMS States in {file_path}: {unique_states}")
# Check if battery loads correctly

# Create the primary axis
fig, ax1 = plt.subplots()

# Plot the primary y-axis data (load)
ax1.plot(load[start:stop], label='Load', color='blue')
ax1.plot(power[start:stop], label='PV_Power', color='orange')
ax1.set_xlabel('Time')
ax1.set_ylabel('Load', color='blue')
ax1.tick_params('y', colors='blue')

# Create the secondary y-axis
ax2 = ax1.twinx()

# Plot the secondary y-axis data (li_ion_capacity)
ax2.plot(li_ion_capacity[start:stop], label='Li-ion Capacity', color='red')
ax2.set_ylabel('Li-ion Capacity', color='red')
ax2.tick_params('y', colors='red')

# Show the legend
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.show()


# print(np.where(EMS_State == 0))
# print(np.where(EMS_State==2.0))
# plt.plot(li_ion_capacity[start:stop])
# indices = np.where((li_ion_capacity > 0.2) & (net_power < 0) & (SoCH2 > 0) & (battery_power < net_power))[0]
# print(f'Indices where conditions are met: {indices}')
# print(f'EMS state at {start} is {EMS_State[start]}')
# print(f'SoCH2 at {start} is {SoCH2[start]}')
i = start
# plt.plot(battery_power[start:stop])
# print(load[i]-power[i]+soec_power[i]-sofc_power[i]-battery_power[i]-deficit_energy[i]+loss_energy[i])
# print(li_ion_capacity[i])
plt.savefig("out.pdf")
