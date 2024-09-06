# -*- coding: utf-8 -*-
import numpy as np
import os
import pandas as pd
from devices.global_constants import *
import devices.photovoltaic as pv
import devices.soe as soe
import devices.sofc as sofc
import devices.bess as bess
import ems


# Load the load, irradiance and temperature data from an excel sheet
df_weather = pd.read_excel('load.xlsx', index_col=0, usecols='A, C:E, T')
df_weather['IRRAD'] = df_weather['IDIFF_H'] + df_weather['IBEAM_H']
case = df_weather[["IRRAD", "TAMB"]].to_numpy()
time = df_weather.index.to_numpy()
load = df_weather['E_el_HH'].to_numpy()

# Convert units - from kWh to Wh
load = load * 1000

# Simulate photovoltaic power generation
pvg = pv.PhotovoltaicPanel(parameters)

# Set the initial State of Charge of the Energy Storage System
SOC_INITIAL_H2 = 1.0
SOC_INITIAL_BESS = 0.2


# Simulate the Hybrid Energy System Operation
def f(n_pv, n_bat, n_cells, n_tank):
    capacityh2 = n_tank * H2_tank_capacity  # [dm^3]
    ub = 1280 * n_bat  # battery energy rating [Wh] * time
    qb = ub  # Assuming the battery power rating = energy rating[W]

    print(f'Updated parameters: Battery energy rating: {ub}, BESS power rating {qb}, \n'
          f'{n_pv} PV modules, The hydrogen tanks can store {n_tank * 7000} litres of H2 in MyH2 7000l tanks, \n'
          f'Solid oxide stack has {n_cells} cells')

    power = np.zeros(len(load))
    
    # Load generated pv power and load profile
    for idx, (i, j) in enumerate(case):
        if i == 0:
            power[idx] = 0
        else:
            power[idx] = n_pv * pvg.photovoltaic_power([(i,j)])        
    net_power = power - load
    
    # Initialize all of the arrays for tracking the system parameters over the year
    lion_capacity = np.ones(len(power) + 1) * SOC_INITIAL_BESS
    prodh2 = np.zeros(len(power) + 1) 
    consh2 = np.zeros(len(power) + 1)
    soch2 = np.ones(len(power) + 1) * SOC_INITIAL_H2
    deficit_energy = np.zeros(len(power) + 1) 
    loss_energy = np.zeros(len(power) + 1) 
    state = np.zeros(len(power))
    sofc_power = np.zeros(len(power))
    soec_power = np.zeros(len(power))
    battery_power = np.zeros(len(power))

    # Create instances of components for the energy management system
    bess_bat = bess.Battery()
    sofc_dev = sofc.SolidOxideFuelCell(n_cells)
    soe_dev = soe.SolidOxideElectrolyser(n_cells)
    p_max_fc, j0_fc = sofc_dev.plot_fuel_cell_characteristic() # [W/m^2]


    # Initial state of the system
    j0_fc = j0_fc * 0.7
    p_max_fc = p_max_fc * 0.99

    # Loop through the data (weather and load)
    for i in range(len(power)):
        s = ems.ems(load[i], power[i], lion_capacity[i], qb, soch2[i], state[i], ub)
        if s == 0: input()

        if s == 1:  # BESS charging, rSOC turned off
            charge_capacity = -bess_bat.bess_charge_capacity(qb, 
                                bc, ub, lion_capacity[i], SOC_max, 1)
            charge_capacity = np.minimum(float(charge_capacity), 
                                         float(net_power[i]))           
            dsoc = charge_capacity / (ub)
            dsoch2 = 0 / capacityh2
            deficit_energy[i] = 0
            loss_energy[i] = abs(net_power[i]) - charge_capacity
            sofc_power[i] = 0
            soec_power[i] = 0
            battery_power[i] = -charge_capacity 
             
        elif s == 2:  # rSOC producing hydrogen, BESS turned of
            soec_power_max = net_power[i]
            hydrogen_possible_production = soe_dev.hydrogen_production_rate(soe_dev.newton_method(
                    soe_dev.w_soec_diff, t, j0_fc, 115000, soec_power_max)) * 22.4 * 3600
            prodh2[i] = np.minimum(hydrogen_possible_production, capacityh2 - soch2[i] * capacityh2)
            dsoch2 = prodh2[i] / capacityh2
            dsoc = 0 / ub
            soec_power[i] = soec_power_max * prodh2[i] / hydrogen_possible_production
            deficit_energy[i] = 0
            loss_energy[i] = abs(net_power[i] - soec_power[i])
            sofc_power[i] = 0
            battery_power[i] = 0
            
        elif s == 3:  # BESS discharging, rSOC turned off
            discharge_capacity = bess_bat.bess_discharge_capacity(
                qb, bd, ub, lion_capacity[i], SOC_min, 1)
            discharge_capacity = np.minimum(float(discharge_capacity), 
                                            float(-net_power[i]))
            dsoc = - discharge_capacity / (ub)
            dsoch2 = 0 / capacityh2
            deficit_energy[i] = abs(net_power[i]) - discharge_capacity
            loss_energy[i] = 0
            sofc_power[i] = 0
            soec_power[i] = 0
            battery_power[i] = discharge_capacity 
                   
        elif s == 4: # rSOC producing electricity, BESS turned off
            p_fc = min(p_max_fc * n_cells * a_cell, abs(net_power[i]))
            j0 = j0_fc # point ahead of p_max
            hydrogen_required = abs(sofc_dev.hydrogen_consumption_rate(
                sofc_dev.newton_method(sofc_dev.w_sofc_diff, t, j0, 115000, p_fc))) * 22.4 * 3600
            consh2[i] = np.minimum(hydrogen_required, soch2[i] * capacityh2)
            dsoc = 0 / ub
            dsoch2 = -consh2[i] / capacityh2
            loss_energy[i] = 0
            sofc_power[i] = p_fc * consh2[i] / hydrogen_required
            deficit_energy[i] = abs(net_power[i]) - sofc_power[i]
            soec_power[i] = 0
            battery_power[i] = 0

        elif s == 5:  # rSOC producing electricity, BESS discharging
            discharge_capacity = bess_bat.bess_discharge_capacity(
                qb, bd, ub, lion_capacity[i], SOC_min, 1)
            discharge_capacity = np.minimum(float(discharge_capacity), 
                                            float(-net_power[i]))
            net = net_power[i] + discharge_capacity
            p_fc = min(p_max_fc * n_cells * a_cell, abs(net))
            j0 = j0_fc # point ahead of p_max [mA/cm2]
            dsoc = - discharge_capacity / (ub)
            hydrogen_required = abs(sofc_dev.hydrogen_consumption_rate(
                sofc_dev.newton_method(sofc_dev.w_sofc_diff, t, j0, 115000, p_fc))) * 22.4 * 3600
            consh2[i] = np.minimum(hydrogen_required, soch2[i] * capacityh2)
            dsoch2 = -consh2[i] / capacityh2
            sofc_power[i] = p_fc * consh2[i] / hydrogen_required
            battery_power[i] = discharge_capacity 
            deficit_energy[i] = abs(net_power[i]) - sofc_power[i] - battery_power[i]
            loss_energy[i] = 0
            soec_power[i] = 0

        elif s == 6:  # rSOC producing hydrogen, BESS charging
            charge_capacity = -bess_bat.bess_charge_capacity(qb,
                                bc, ub, lion_capacity[i], SOC_max, 1)
            soec_power_max = net_power[i] - charge_capacity
            j0 = j0_fc
            hydrogen_possible_production = soe_dev.hydrogen_production_rate(soe_dev.newton_method(
                    soe_dev.w_soec_diff, t, j0, 115000, soec_power_max)) * 22.4 * 3600
            dsoc = charge_capacity / (ub)
            prodh2[i] = np.minimum(hydrogen_possible_production, capacityh2 - soch2[i] * capacityh2)
            dsoch2 = prodh2[i] / capacityh2
            soec_power[i] = soec_power_max * prodh2[i]/hydrogen_possible_production
            deficit_energy[i] = 0
            loss_energy[i] = 0
            sofc_power[i] = 0
            battery_power[i] = charge_capacity
            
        elif s == 7:  # empty storage system — deficit of electricity
            deficit_energy[i] = load[i] - power[i]
            dsoc = 0 / ub
            loss_energy[i] = 0
            dsoch2 = 0 / capacityh2
            sofc_power[i] = 0
            soec_power[i] = 0
            battery_power[i] = 0
            
        elif s == 8:  # full storage system — produced electricity is los
            deficit_energy[i] = 0
            loss_energy[i] = power[i] - load[i]
            dsoc = 0 / ub
            dsoch2 = 0 / capacityh2
            sofc_power[i] = 0
            soec_power[i] = 0
            battery_power[i] = 0

        else:
            pass

        lion_capacity[i+1] = lion_capacity[i] + dsoc
        soch2[i+1] = soch2[i] + dsoch2
        state[i] = s

    time_csv = np.array(time, dtype='datetime64')
    result = np.column_stack((np.array(lion_capacity[:-1]), np.array(power), np.array(sofc_power),
                              np.array(soec_power), np.array(battery_power),np.array(load), 
                              np.array(soch2[:-1]),np.array(state),np.array(net_power),
                              np.array(deficit_energy[:-1]),np.array(loss_energy[:-1])))

    return time_csv, result

if __name__ == "__main__":
    # Path to the file with results
    output = 'testing_results'
    os.makedirs(output, exist_ok=True)
    text_csv_filename = os.path.join(output, '1.csv')

    # State the size of the system
    time_csv, result = f(12, 28, 30, 55)

    # Print the cumulative sum of energy deficit and load
    cumulative_sum_columns = np.sum(result[:-1, -2:], axis=0)
    print(cumulative_sum_columns)
    np.savetxt(text_csv_filename, result, header='li_ion_capacity PV_power SOFC_power SOEC_power battery_power '
                                                 'load SoCH2 EMS_State net_power energy_deficit energy_loss')

