import math

import numpy as np
from composabl import Teacher

class Teacher(Teacher):
    def __init__(self, *args, **kwargs):
        self.obs_history = None
        self.reward_history = []
        self.last_reward = 0
        self.error_history = []
        self.rms_history = []
        self.last_reward = 0
        self.count = 0

    async def transform_sensors(self, obs, action):
        return obs

    async def transform_action(self, transformed_obs, action):
        return action

    async def filtered_sensor_space(self):
        return ['T', 'Tc', 'Ca', 'Cref', 'Tref','Conc_Error', 'Eps_Yield', 'Cb_Prod']

    async def compute_reward(self, transformed_obs, action, sim_reward):
        if self.obs_history is None:
            self.obs_history = [transformed_obs]
            return 0.0
        else:
            self.obs_history.append(transformed_obs)

        error = (float(transformed_obs['Cref']) - float(transformed_obs['Ca']))**2
        self.error_history.append(error)
        rms = math.sqrt(np.mean(self.error_history))
        self.rms_history.append(rms)

        reward = math.exp(-0.01*np.sum(self.error_history))

        self.reward_history.append(reward)

        self.count += 1
        return reward

    async def compute_action_mask(self, transformed_obs, action):
        return None

    async def compute_success_criteria(self, transformed_obs, action):
        success = False
        return success

    async def compute_termination(self, transformed_obs, action):
        return False
