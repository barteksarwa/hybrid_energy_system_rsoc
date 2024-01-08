# -*- coding: utf-8 -*-
"""
2023

@author: Bartlomiej Sarwa
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
import pandas as pd
from devices.global_constants import *
import devices.photovoltaic as pv
import devices.soe as soe
import devices.sofc as sofc
import devices.bess as bess
import ems
import math


# Load the data (load, irradiance and temperature) from an excel sheet
df_weather = pd.read_excel('load.xlsx', index_col=0, usecols='A, C:E, T')
df_weather['IRRAD'] = df_weather['IDIFF_H'] + df_weather['IBEAM_H']
case = df_weather[["IRRAD", "TAMB"]].to_numpy()
time = df_weather.index.to_numpy()
load = df_weather['E_el_HH'].to_numpy()
load = load * 1000  # kWh to Wh
# Simulate photovoltaic power generation
pvg = pv.PhotovoltaicPanel(parameters)

def f(n_pv, n_bat, n_cells_f, n_tank):
    # Overwrite parameters of the system
    parameters['N_s']= n_pv * 60  # number of pv modules * number of cells in one module
    capacityh2 = n_tank * 7000 # litres
    qb = 1280 * n_bat # battery energy rating * number of batteries
    ub = 1280 * n_bat
    n_cells = 1 * n_cells_f # number of cells * number of SOFC units
    print(f'Updated parameters: Battery energy rating: {ub}, BESS power rating {qb}, \n'
          'Number of PV modules {n_pv}')
    power = np.zeros(len(load))
    
    # Load generated pv power and load profile
    for idx, (i, j) in enumerate(case):
        if i == 0:
            power[idx] = 0
        else:
            power[idx] = n_pv * pvg.photovoltaic_power([(i,j)])        
    net_power = power - load
    
    # Initialize all of the arrays for tracking the system parameters over the year
    lion_capacity = np.ones(len(power) + 1) * 0.75
    prodh2 = np.zeros(len(power) + 1) 
    consh2 = np.zeros(len(power) + 1)
    soch2 = np.ones(len(power) + 1) * 0.98
    deficit_energy = np.zeros(len(power) + 1) 
    loss_energy = np.zeros(len(power) + 1) 
    state = np.zeros(len(power) + 1)
    sofc_power = np.zeros(len(power))
    soec_power = np.zeros(len(power))
    battery_power = np.zeros(len(power))

    bess_bat = bess.Battery()
    sofc_dev = sofc.SolidOxideFuelCell()
    soe_dev = soe.SolidOxideElectrolyser()
    
    # Initial state of the system
    state[0] = 4
    
    # Simulation of the operation of the system
    for i in range(len(power)):
        s = ems.ems(load[i], power[i], lion_capacity[i], qb, soch2[i], state[i])

        if s == 1:  # charge battery
            charge_capacity = -bess_bat.bess_charge_capacity(qb, 
                                bc, ub, lion_capacity[i], SOC_max, 1) * n_bat
            charge_capacity = np.minimum(float(charge_capacity), 
                                         float(net_power[i]))           
            dsoc = charge_capacity / ub
            dsoch2 = 0 / capacityh2
            deficit_energy[i] = 0
            loss_energy[i] = abs(net_power[i])-charge_capacity
            sofc_power[i] = 0
            soec_power[i] = 0
            battery_power[i] = -charge_capacity * n_bat
             
        elif s == 2:  # produce hydrogen in SOEC
            soec_power_max = net_power[i] # change to include maximum reasonable SOEC power as min(net_power, soec_max)
            hydrogen_possible_production = soe_dev.hydrogen_production_rate(soe_dev.newton_method(
                    soe_dev.w_soec_diff, 1100, 200, 115000, 5, soec_power_max)) * 22.4 * 3600
            prodh2[i] = np.minimum(hydrogen_possible_production, capacityh2 - soch2[i] * capacityh2)
            dsoch2 = prodh2[i] / capacityh2
            dsoc = 0 / ub
            soec_power[i] = soec_power_max * prodh2[i]/hydrogen_possible_production
            deficit_energy[i] = 0
            loss_energy[i] = abs(net_power[i] - soec_power[i])
            sofc_power[i] = 0
            battery_power[i] = 0
            
        elif s == 3:  # draw power from battery
            discharge_capacity = bess_bat.bess_discharge_capacity(
                qb, bd, ub, lion_capacity[i], SOC_min, 1) * n_bat
            discharge_capacity = np.minimum(float(discharge_capacity), 
                                            float(-net_power[i]))
            dsoc = - discharge_capacity / ub
            dsoch2 = 0 / capacityh2
            deficit_energy[i] = abs(net_power[i])-discharge_capacity
            loss_energy[i] = 0
            sofc_power[i] = 0
            soec_power[i] = 0
            battery_power[i] = discharge_capacity * n_bat
                   
        elif s == 4: # generate power in fuel cell
            p_fc = min(850 * n_cells * a_cell, abs(net_power[i]))
            # if net_power[i] > 100:
            j0 = 1000 # punkt przed p_max
            hydrogen_required = abs(sofc_dev.hydrogen_consumption_rate(
                sofc_dev.newton_method(sofc_dev.w_sofc_diff, 1100, j0, 115000, p_fc, 100))) * 22.4 * 3600
            consh2[i] = np.minimum(hydrogen_required, soch2[i] * capacityh2)

            dsoc = 0 / ub
            dsoch2 = -consh2[i] / capacityh2
            loss_energy[i] = 0
            sofc_power[i] = p_fc*consh2[i]/hydrogen_required
            deficit_energy[i] = abs(net_power[i]) - sofc_power[i]
            soec_power[i] = 0
            battery_power[i] = 0

        elif s == 5:  # draw power from battery and generate power in fuel cell
            discharge_capacity = bess_bat.bess_discharge_capacity(
                qb, bd, ub, lion_capacity[i], SOC_min, 1) * n_bat
            discharge_capacity = np.minimum(float(discharge_capacity), 
                                            float(-net_power[i]))
            net = net_power[i]+discharge_capacity
            p_fc = min(850 * n_cells * a_cell, abs(net))
            j0 = 1000 # punkt przed p_max 
            dsoc = - discharge_capacity / ub
            hydrogen_required =abs(sofc_dev.hydrogen_consumption_rate(
                sofc_dev.newton_method(sofc_dev.w_sofc_diff, 1100, j0, 115000, p_fc, 100))) * 22.4 * 3600
            consh2[i] = np.minimum(hydrogen_required, soch2[i] * capacityh2)
            dsoch2 = -consh2[i] / capacityh2
            sofc_power[i] = p_fc * consh2[i] / hydrogen_required
            battery_power[i] = discharge_capacity * n_bat
            deficit_energy[i] = abs(net_power[i]) - sofc_power[i]-battery_power[i]
            loss_energy[i] = 0
            soec_power[i] = 0

            
        elif s == 6:  # energy deficit due to lack of loaded storage system
            deficit_energy[i] = load[i] - power[i]
            dsoc = 0 / ub
            loss_energy[i] = 0
            dsoch2 = 0 / capacityh2
            sofc_power[i] = 0
            soec_power[i] = 0
            battery_power[i] = 0
            
        elif s == 7:  # energy loss due to max capacity of energy storage
            deficit_energy[i] = 0
            loss_energy[i] = power[i] - load[i]
            dsoc = 0 / ub
            dsoch2 = 0 / capacityh2
            sofc_power[i] = 0
            soec_power[i] = 0
            battery_power[i] = 0
            
        elif s == 8:  # produce hydrogen in SOEC and charge battery
            charge_capacity = -bess_bat.bess_charge_capacity(qb, 
                                bc, ub, lion_capacity[i], SOC_max, 1) * n_bat    
            soec_power_max = net_power[i] - charge_capacity # change to include maximum reasonable SOEC power as min(net_power, soec_max)
            hydrogen_possible_production = soe_dev.hydrogen_production_rate(soe_dev.newton_method(
                    soe_dev.w_soec_diff, 1100, 200, 115000, 5, soec_power_max)) * 22.4 * 3600
            dsoc = charge_capacity / ub
            prodh2[i] = np.minimum(hydrogen_possible_production, capacityh2 - soch2[i] * capacityh2)
            dsoch2 = prodh2[i] / capacityh2
            soec_power[i] = soec_power_max * prodh2[i]/hydrogen_possible_production
            deficit_energy[i] = 0
            loss_energy[i] = 0
            sofc_power[i] = 0
            battery_power[i] = charge_capacity    
        
        else:
            pass
            if i < 1:  # print initial parameters
                print(discharge_capacity, dsoc, qb, 
                      bd, ub, lion_capacity[i], g_max)
                
        lion_capacity[i+1] = lion_capacity[i] + dsoc
        soch2[i+1] = soch2[i] + dsoch2
        state[i] = s
    
    # for i in range(len(lion_capacity)):
    #     if lion_capacity[i] < 0.2:
    #         lion_capacity[i] = 0.2
            
    # for i in range(len(soch2)):
    #     if soch2[i] > 1.0:
    #         soch2[i] = 1.0
    #     elif soch2[i] < 0:
    #         soch2[i] = 0
            
    # for i in range(len(consh2)):
    #     if consh2[i] < 0.0:
    #         consh2[i] = 0.0
    
    producedh2 = np.cumsum(prodh2)
    consumedh2 = np.cumsum(consh2)
    
    time_csv = np.array(time, dtype='datetime64')
    result = np.column_stack((np.array(lion_capacity[:-1]), np.array(power), np.array(sofc_power),
                              np.array(soec_power), np.array(battery_power),np.array(load), 
                              np.array(soch2[:-1]),np.array(state[:-1]),np.array(net_power),
                              np.array(deficit_energy[:-1]),np.array(loss_energy[:-1])))

    return time_csv, result

output = 'test_plots_csv'
os.makedirs(output, exist_ok=True)
        
text_csv_filename = os.path.join(output, 'test.csv')
time_csv, result = f(12,1,16,5)
cumulative_sum_columns = np.sum(result[:, -2:], axis=0)
print(cumulative_sum_columns)
np.savetxt(text_csv_filename, result, header='li_ion_capacity PV_power SOFC_power SOEC_power battery_power load SoCH2 EMS_State net_power energy_deficit energy_loss')

# output_directory = 'doe_output_csv'
# os.makedirs(output_directory, exist_ok=True)

# df_designs = pd.read_excel('denormalized_designs.xlsx')
# design_table = df_designs.to_numpy()

# for i, row in enumerate(design_table, start=1):
#     time_csv, output = f(*row)

#     # Create file names with iteration number
#     text_csv_filename = os.path.join(output_directory, f'pv_plot_{i}.csv')
#     time_csv_filename = os.path.join(output_directory, f'time.csv')

#     # Save files
#     np.savetxt(text_csv_filename, output)
#     if i == 1:  # Save 'time' only for the first iteration
#         np.savetxt(time_csv_filename, time_csv)
        


        