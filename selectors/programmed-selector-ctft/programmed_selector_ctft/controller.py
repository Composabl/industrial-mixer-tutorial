# Copyright (C) Composabl, Inc - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

from typing import Dict, List

from composabl_core import SkillController

class Controller(SkillController):
    def __init__(self, *args, **kwargs):
        self.counter = 0

    async def compute_action(self, obs, action):
        self.counter += 1
        # Error greater than 10%
        if (abs(float(obs['Ca']) - float(obs['Cref'])) /float(obs['Cref']) ) >= 0.1:
            action = 1
        else:
            action = 0

        return action

    async def transform_sensors(self, obs):
        return obs

    async def filtered_sensor_space(self):
        return ['T', 'Tc', 'Ca', 'Cref', 'Tref','Conc_Error', 'Eps_Yield', 'Cb_Prod']

    async def compute_success_criteria(self, transformed_obs, action):
        return False

    async def compute_termination(self, transformed_obs, action):
        return False

