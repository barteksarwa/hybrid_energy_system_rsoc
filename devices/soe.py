import numpy as np
from .global_constants import *

class SolidOxideElectrolyser:
    def __init__(self):
        pass
    
    @staticmethod
    def partial_pressure(p, x):
        return p * x

    def equilibrium_voltage(self, t, p):
        # v_0n = (-(water_enthalpy_820 - t*water_specific_entropy_820)+(hydrogen_enthalpy_820 - t*hydrogen_specific_entropy_820)+0.5*(oxygen_enthalpy_820-t*oxygen_specific_entropy_820))/n_e/F
        v_0n = 1.253 - 2.4516 * t * 0.0001
        p_h2 = self.partial_pressure(p, x_h2)
        p_o2 = self.partial_pressure(p, x_o2)
        p_h2o = self.partial_pressure(p, x_h2o)
        return v_0n + R * t / n_e / F * np.log(p_h2 * p_o2 / p_h2o)

    @staticmethod
    def j_0a(t):
        return g_a * np.exp(-e_acta / R / t)

    @staticmethod
    def j_0c(t):
        return g_c * np.exp(-e_actc / R / t)

    def v_acta(self, t, j):
        return R * t * np.arcsinh(j / 2 / self.j_0a(t)) / n_e / F

    def v_actc(self, t, j):
        return R * t * np.arcsinh(j / 2 / self.j_0c(t)) / n_e / F

    @staticmethod
    def v_ohm(t, j):
        return j * (a_a * sigma_a * np.exp(b_a / t) + a_c *
                  sigma_c * np.exp(b_c / t) + a_e * sigma_e * np.exp(b_e / t))

    @staticmethod
    def binary_diffusion_coefficient_anode(t, p):
        return 0.00143 * t**1.75 / (p * np.sqrt(m_h2o_h2) * \
                                ((sigma_f_h2o)**1/3 + (sigma_f_h2)**1/3)**2)

    def binary_diffusion_coefficient_cath(self, t, p):
        return 0.00143 * t**1.75 / (p * np.sqrt(m_n2_o2) *
                                ((sigma_f_o2)**1/3 + (sigma_f_n2)**1/3)**2)

    @staticmethod
    def knudsen_h2o(t):
        return 4 / 3 * r_pore * np.sqrt(8 * R * t / np.pi * m_h2o)

    @staticmethod
    def knudsen_o2(t):
        return 4 / 3 * r_pore * np.sqrt(8 * R * t / np.pi * m_o2)
    
    @staticmethod
    def knudsen_h2(t):
        return 4 / 3 * r_pore * np.sqrt(8 * R * t / np.pi * m_h2)
    
    def eff_diff_steam(self, t, p):
        return electrode_porosity / electrode_tortuosity / \
            (1 / self.knudsen_h2o(t) + 1 / 
             self.binary_diffusion_coefficient_anode(t, p))

    def eff_diff_oxygen(self, t, p):
        return electrode_porosity / electrode_tortuosity / \
            (1 / self.knudsen_o2(t) + 1 / self.binary_diffusion_coefficient_cath(t, p)) # poprawic

    def eff_diff_hydrogen(self, t, p):
        return electrode_porosity / electrode_tortuosity / \
            (1 / self.knudsen_h2(t) + 1 / self.binary_diffusion_coefficient_cath(t, p)) # poprawic

    def v_conca(self, t, j, p):
        p_h2o = self.partial_pressure(p, x_h2o)
        p_h2 = self.partial_pressure(p, x_h2)   
        return R * t / n_e / F * np.log\
            ((1 + j * R * t * sigma_a / 2 / F / self.eff_diff_steam(t, p) / p_h2)/
            (1 - j * R * t * sigma_a / 2 / F / self.eff_diff_steam(t, p) / p_h2o))

    # def v_concc(self, t, j, p):
    #     # p_o2 = self.partial_pressure(p, x_o2)
    #     # return R * t / n_e / F * np.log((1 + j * R * t * sigma_c /
    #     #         2 / F / self.eff_diff_oxygen(t, p) / p_o2)**.5)
    #     return 0

    def first_principle_model(self, t, j, p):
        j0a = self.j_0a(t)
        j0c = self.j_0c(t)
        p_h2 = self.partial_pressure(p, x_h2)
        p_o2 = self.partial_pressure(p, x_o2)
        p_h2o = self.partial_pressure(p, x_h2o)
        v_n = self.equilibrium_voltage(t, p)
        v_c = v_n + self.v_acta(t, j) + self.v_actc(t, j) + self.v_ohm(t, j) +\
             self.v_conca(t, j, p) #+ self.v_concc(t, j, p)
        return v_c

    # def collision_integral(t):
    #     t_ih2 = t/t_refh2

    @staticmethod
    def p_soec(v_c, j, w_sc):
        return v_c * j * a_cell * n_cells + w_sc

    @staticmethod
    def w_sc(s):
        if s in range(3, 6, 1):
            return 1
        else:
            return 0

    def w_soec(self, t, j, p, w_sc):
        return self.first_principle_model(t, j, p) \
            * j * a_cell * n_cells + w_sc

    def w_soec_diff(self, t, j, p, w_sc, w_0):
        return w_0 - self.w_soec(t, j, p, w_sc)

    def s_gen(self, t, j):
        v_conc = self.v_conca(t, j, p) + self.v_concc(t, j, p)
        v_act = self.v_acta(t, j) + self.v_actc(t, j)
        return n_e * F * (v_act + self.v_ohm(t, j) + v_conc) / t
    
    def q_soec(self, t, s_gen, s_in, s_out):
        return -t * self.s_gen(t, j) - t * (s_in - s_out)

    def e_th_soec(self, t, q_soec):
        return self.q_soec(t, s_gen, s_in, s_out) * (1 - t_0 / t)

    @staticmethod
    def central_difference_quotient(f, t, j, p, w_sc, w_0, h=1e-6):
        return (f(t, j + h, p, w_sc ,w_0) \
                - f(t, j - h, p, w_sc, w_0)) / (2 * h)

    def newton_method(self, f, t, j, p, w_sc, w_0, epsilon=1e-6, max_iter=1000):
        for i in range(max_iter):
            wj = f(t, j, p, w_sc, w_0)
            if abs(wj) < epsilon:
                return j
            dwj = self.central_difference_quotient(f, t, j, p, w_sc, w_0)
            if dwj == 0:
                break
            j = j - wj / dwj
        return j

    @staticmethod
    def hydrogen_production_rate(j):
        i = j * a_cell * n_cells
        return i * coulomb / avogadro_number / 2
