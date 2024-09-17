# Copyright (C) Composabl, Inc - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import random
from typing import Dict, List

from composabl_core import SkillController

import numpy as np

def PID(TPV,
        TSP,
        Kp,
        TauI,
        TauD,
        Tkm1,
        dt = 1,
        Ubias = 0,
        I = 0,
        N = 10,
        b = 1,
        c = 0,
        Method = 'Backward'):

    e = TSP - TPV
    e_before = TSP - Tkm1
    #print('erros: ',e, e_before)
    Ti = TauI
    Td = TauD

    if Method == 'Backward':
        b1 = Kp * dt / Ti if Ti != 0 else 0.0
        b2 = 0.0
        ad = Td / (Td + N * dt)
        bd = Kp * Td * N / (Td + N * dt)

    elif Method == 'Forward':
        b1 = 0.0
        b2 = Kp * dt / Ti  if Ti != 0 else 0.0
        ad = 1 - N * dt / Td if Td != 0 else 0.0
        bd = Kp * N

    elif Method == 'Tustin':
        b1 = Kp * dt / 2 / Ti if Ti != 0 else 0.0
        b2 = b1
        ad = (2 * Td - N * dt) / (2 * Td + N * dt)
        bd = 2 * Kp * Td * N / (2 * Td + N * dt)

    elif Method == 'Ramp':
        b1 = Kp * dt / 2 / Ti if Ti != 0 else 0.0
        b2 = b1
        ad = np.exp(-N * dt / Td) if Td != 0 else 0.0
        bd = Kp * Td * (1 - ad) / dt

    P = e
    P = Kp * (b * e)
    # Integral action:
    I = e*dt + I
    I = b1 * (e) + b2 * (e_before)
    # Derivative Action:
    D = (e-e_before)/dt
    D_int = 0
    D  = ad * D_int + bd * ((c * e) - (c * e_before))

    #U = U0 + Kp*(P + (1/TauI)*I + TauD*D)
    #U = U0 + Kp + Ki*I + Kd*D
    U = Ubias + P + I + D

    return U


class Controller(SkillController):
    def __init__(self, *args, **kwargs):
        self.count = 0
        self.I = 0
        self.T_list = []
        self.ΔTc = 0

    async def compute_action(self, obs, action):
        #print("OBS: ", obs)
        self.Tref = float(obs['Tref'])
        self.T = float(obs['T'])
        bias = 0

        if self.count > 0:
            self.ΔTc = PID(self.T,TSP=self.Tref,Kp=0.04,
                        TauI=1.8,TauD=1.5,
                        Tkm1=self.T_list[-1],dt=1,Ubias=bias, I=self.I, N=1, Method='Backward')

        self.T_list.append(self.T)

        self.count += 1
        return [self.ΔTc]

    async def transform_sensors(self, obs):
        return obs

    async def filtered_sensor_space(self):
        return ['T', 'Tc', 'Ca', 'Cref', 'Tref','Conc_Error', 'Eps_Yield', 'Cb_Prod']

    async def compute_success_criteria(self, transformed_obs, action):
        return False

    async def compute_termination(self, transformed_obs, action):
        return False



