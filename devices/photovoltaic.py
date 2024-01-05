import numpy as np
from pvlib import pvsystem
from .global_constants import *


class PhotovoltaicPanel:
    def __init__(self, parameters):
        self.parameters = parameters
        self.v_mp = []
        self.i_mp = []

    def photovoltaic_power(self, case):
        for i, j in case:
            IL, I0, Rs, Rsh, nNsVth = pvsystem.calcparams_desoto(
                i,
                j,
                alpha_sc=parameters['alpha_sc'],
                a_ref=parameters['a_ref'],
                I_L_ref=parameters['I_L_ref'],
                I_o_ref=parameters['I_o_ref'],
                R_sh_ref=parameters['R_sh_ref'],
                R_s=parameters['R_s'],
                EgRef=1.121,
                dEgdT=-0.0002677
                )
            curve_info = pvsystem.singlediode(
                    photocurrent=IL,
                    saturation_current=I0,
                    resistance_series=Rs,
                    resistance_shunt=Rsh,
                    nNsVth=nNsVth,
                    ivcurve_pnts=100,
                    method='lambertw'
                )
            self.v_mp.append(curve_info['v_mp'])
            self.i_mp.append(curve_info['i_mp'])

        self.v_mp = [arr.ravel() for arr in self.v_mp]
        self.v_mp = np.concatenate(self.v_mp)
        self.i_mp = [arr.ravel() for arr in self.i_mp]
        self.i_mp = np.concatenate(self.i_mp)
        power = []

        for x, y in zip(self.i_mp, self.v_mp):
            power.append(x * y * 6)

        return power
