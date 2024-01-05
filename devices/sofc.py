import numpy as np
import sys
sys.path.append("..")
from .global_constants import *


class SolidOxideFuelCell:
    def __init__(self):
        pass
    
    @staticmethod
    def partial_pressure(p, x):
        return p * x

    @staticmethod
    def reaction_rate(j):
        return j / 2 / F

    def equilibrium_voltage(self, t, p):
        p_h2 = self.partial_pressure(p, x_h2)
        p_o2 = self.partial_pressure(p, x_o2)
        p_h2o = self.partial_pressure(p, x_h2o)
        k_eq =  p_h2o
        v_0 = 0.977 + R * t / 2 / F * np.log(p_h2 * (p_o2**0.5) / p_h2o)
        return v_0

    @staticmethod
    def activation_loss(j):
        return 0.03 * np.log(j / j_0fc) # V

    @staticmethod
    def ohmic_loss(t, j):
        r = []
        for i, k, m in zip(ro_i_const, ro_i_exp, delta_i):
            r.append(i * m*10 * np.exp(k / t)) # ohmcm^2
        return j * sum(r) * 0.0001  # A/m^2 * Ohmcm^2, unit adjustment
    
    @staticmethod
    def concentration_loss(j):
        if j <= 0 or j >= j_lfc: # A/m^2
            # Handle invalid values to avoid log of non-positive number
            return 0  # or another appropriate value
        else:
            return -0.08 * np.log(1 - j / j_lfc) # V

    def first_principle_model(self, t, j, p):
        p_h2 = self.partial_pressure(p, x_h2)
        p_o2 = self.partial_pressure(p, x_o2)
        p_h2o = self.partial_pressure(p, x_h2o) 
        v_n = self.equilibrium_voltage(t, p)
        v_loss = self.activation_loss(j) + self.ohmic_loss(t, j) \
            + self.concentration_loss(j)
        v_c = v_n - v_loss
        return v_c
    
    def power_sofc(self, t, p):
        i0 = np.linspace(0, 3900, 100)
        e = []
        power = []
        for i in i0:
            e.append(self.first_principle_model(t, i, p))
        for i, k in zip(i0, e):
            power.append(i * k)
        return i0, power
    
    def w_sofc(self, t, j, p):
        return self.first_principle_model(t, j, p) * j * a_cell_fc * n_cells_fc

    def w_sofc_diff(self, t, j, p, w_0):
        return w_0 - self.w_sofc(t, j, p)

    @staticmethod
    def central_difference_quotient(f, t, j, p, w_0, h=1e-6):
        return (f(t, j + h, p, w_0) - f(t, j - h, p, w_0)) / (2 * h)

    def newton_method(self, f, t, j, p, w_0, epsilon=1e-6, max_iterations=1000):
        for i in range(max_iterations):
            wj = f(t, j, p, w_0)
            if abs(wj) < epsilon:
                return j
            dwj = self.central_difference_quotient(f, t, j, p, w_0)
            if dwj == 0:
                break
            j = max(j - wj / dwj, 10)
        return j

    @staticmethod
    def hydrogen_consumption_rate(j):
        i = j * a_cell_fc * n_cells_fc
        return i * coulomb / avogadro_number / 2

    def plot_sofc(self, t, p, i0):
        max_p = np.nanmax(p)
        max_index = np.nanargmax(p)
        import matplotlib.pyplot as plt
        plt.plot(i0, p, '-b', markersize=5)
        plt.plot(i0[max_index], max_p, 'bo', markersize=8)
        plt.xlabel("j (A/m2)")
        plt.ylabel("P (W/m2)")
        plt.title('Fuel cell characteristic')
        plt.show()

    def plot_i_v(self, t, i, p):
        import matplotlib.pyplot as plt
        plt.plot(i0, v_c_values, '-b', markersize=5)
        plt.xlabel("Current Density (A/m²)")
        plt.ylabel("Cell Voltage (V)")
        plt.title('Fuel Cell I(V) Characteristic')
        plt.show()
        
# # # Create an instance of SolidOxideFuelCell
# sofc_ins = SolidOxideFuelCell()

# # # Get power values
# i0, power = sofc_ins.power_sofc(1100, 115000)
# # Calculate cell voltages using the first-principle model
# v_c_values = [sofc_ins.first_principle_model(1100, current_density, 115000) for current_density in i0]


# # # Plot the results
# sofc_ins.plot_sofc(1100, power, i0)
# print(np.nanmax(power))
# sofc_ins.plot_i_v(1100, v_c_values, i0)


# ## Tests 
# print(sofc_ins.ohmic_loss(1100, 500))



      