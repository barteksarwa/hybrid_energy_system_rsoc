# Scientific constants
R = 8.314  # Universal gas constant [J/mol K]
F = 9.64853321233e4  # Faraday constant [C/mol]
avogadro_number = 6.022e23  # mol-1
coulomb = 6.241509e18
t = 1173 # Kelvin

# Thermodynamics properties of H2, H2O, O2
hydrogen_specific_entropy_820 = 18.19  # J/(g·K)
hydrogen_enthalpy_820 = 42.19  # kJ/mol
oxygen_specific_entropy_820 = 32.29  # J/(g·K)
oxygen_enthalpy_820 = 33.06  # kJ/mol
water_specific_entropy_820 = 37.54  # J/(g·K)
water_enthalpy_820 = -240.99  # kJ/mol
n_e = 2

# Solid oxide electrolyser electrochemical properties
g_a = 2.051e9  # A/m2 preexponential factor for anode
g_c = 1.344e10  # A/m2 preexponential factor for cathode
e_actc = 1e5  # J/mol
e_acta = 1.2e5  # J/mol
sigma_a = 5e-5  # m
sigma_c = 5e-5 # m 
sigma_e = 20e-5 # m
a_a = 2.98e-5  # no unit
a_c = 8.11e-5  # no unit
a_e = 2.94e-5  # no unit
b_a = -1392  # no unit
b_c = 600  # no unit
b_e = 10350  # no unit
electrode_porosity = 0.48  # no unit
electrode_tortuosity = 5.4  # no unit
m_h2 = 0.002  # molar mass of h2
m_h2o = 0.018  # molar mass of h2o
m_o2 = 0.032  # molar mass of o2
m_n2 = 0.028  # molar mass of n2
m_h2o_h2 = 2/(1/m_h2+1/m_h2o)
m_n2_o2 = 2/(1/m_n2+1/m_o2)
x_h2 = 0.95  # % (molar fraction of Hydrogen inlet of fc)
x_h2o = 1-x_h2  # % (molar fraction of Water inlet of fc)
x_h2_ec = 0.05  # % (molar fraction of Hydrogen inlet of electrolyser)
x_h2o_ec = 1-x_h2_ec  # % (molar fraction of Water inlet of electrolyser)
x_o2 = 0.21  # % (molar fraction of oxygen inlet of electrolyser)
x_n2 = 0.79  # % (molar fraction of nitrogen inlet of electrolyser)
sigma_f_h2 = 6.12  # no unit
sigma_f_h2o = 13.1  # no unit
sigma_f_n2 = 18.5  # no unit
sigma_f_o2 = 16.3  # no unit
r_pore = 0.25e-6  # m
# r_pore from AlZahrani https://sci-hub.se/10.1016/j.ijhydene.2017.03.186

# RSOFC
a_cell = 0.01  # m^2

# Solid Oxide Fuel Cell parameters
# properties of sofc components:
# ro_i_const - material resistivity coeff,
# ro_i_exp - material resisticity exp coeff,
# delta_i - corresponding thickness(cm)
# where i = a, c, e for anode cathode and electrolyte
# j_0fc = 3000  # A/m^2 exchange current density
# j_lfc = 9000  # A/m^2 limiting current density
# ro_i_const = [0.00298, 0.008114, 0.00294]
# ro_i_exp = [-1392, 600, 10350]
# delta_i = [0.5, 0.05, 0.01] # mm
# delta_a = 0.015
# delta_c = 0.2
# delta_e = 0.004
# n_cells_fc = 25
# a_cell_fc = 0.05  # m^2 wpisane zeby dopasowac stos

# Photovoltaic panel parameteres from the datasheet

parameters = {
    'Name': 'Jiangsu Sunport Power Corp. Ltd. SPP320M60B',
    'Date': '03/1/2024',
    'T_NOCT': 43,
    'N_s': 60,
    'I_sc_ref': 10.31,
    'V_oc_ref': 39.6,
    'I_mp_ref': 9.86,
    'V_mp_ref': 32.4,
    'alpha_sc': 0.0062,
    'beta_oc': -0.121,
    'a_ref': 1.428,
    'v_TSTC': 1.38e-23*(25+273.15)/1.602e-19,
    'I_L_ref': 10.36,
    'I_o_ref': 1e-10,
    'R_s': 0.27,
    'R_sh_ref': 570,
    'gamma_r': -0.36,
    'Version': 'MM106',
    'Technology': 'Mono-c-Si',
}

# Li-ion battery parameters

bd = 0.95  # Discharge efficiency [-]
bc = 0.95  # Charge efficiency [-]


# Energy storage system limitations / parameters
SOC_min = 0.2 # Minimum allowable state of charge [-]
SOC_max = 0.8 # Maximum allowable state of charge [-]
SOCH2_min = 0


