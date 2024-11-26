import math
import numpy as np
from composabl import Teacher

class BaseCSTR(Teacher):
    def __init__(self, *args, **kwargs):
        # Initialize history and tracking variables.
        # obs_history: Keeps a record of observed sensor data.
        # reward_history: Stores all the rewards calculated during training.
        # last_reward: Tracks the most recent reward.
        # error_history: Tracks errors between reference concentration (Cref) and actual concentration (Ca).
        # rms_history: Records root mean squared (RMS) error over time.
        # count: Counts the number of times rewards have been computed.
        self.obs_history = None
        self.reward_history = []
        self.last_reward = 0
        self.error_history = []
        self.rms_history = []
        self.count = 0

    # Transforms sensor data if needed. By default, it returns the data unmodified.
    # This can be useful for normalizing or converting sensor values.
    async def transform_sensors(self, obs, action):
        return obs

    # Transforms actions if needed. By default, it returns the action unmodified.
    # This can be useful for converting action formats before execution.
    async def transform_action(self, transformed_obs, action):
        return action

    # Filters the list of sensors to include only those relevant for the skill.
    # In this case, the relevant sensors are T, Tc, Ca, Cref, Tref, Conc_Error, Eps_Yield, and Cb_Prod.
    async def filtered_sensor_space(self):
        return ['T', 'Tc', 'Ca', 'Cref', 'Tref', 'Conc_Error', 'Eps_Yield', 'Cb_Prod']

    # Computes the reward for the agent's actions based on observed sensor data.
    # Uses an exponential decay function to penalize large errors and reward small ones.
    async def compute_reward(self, transformed_obs, action, sim_reward):
        # Initialize observation history if it's the first time this method is called.
        if self.obs_history is None:
            self.obs_history = [transformed_obs]
            return 0.0
        else:
            # Add the current observation to the history.
            self.obs_history.append(transformed_obs)

        # Compute the squared error between the reference concentration (Cref) and the actual concentration (Ca).
        error = (float(transformed_obs['Cref']) - float(transformed_obs['Ca']))**2
        self.error_history.append(error)

        # Compute the root mean square (RMS) error over all errors seen so far.
        rms = math.sqrt(np.mean(self.error_history))
        self.rms_history.append(rms)

        # Compute the reward using an exponential decay based on the cumulative error.
        reward = math.exp(-0.01 * np.sum(self.error_history))
        self.reward_history.append(reward)

        # Increment the count of reward calculations.
        self.count += 1
        return reward

    # Optionally restrict the set of actions available to the agent.
    # Returns None, indicating no restrictions in this case.
    async def compute_action_mask(self, transformed_obs, action):
        return None

    # Defines the success criteria for the skill.
    # This implementation always returns False, indicating no success is defined yet.
    async def compute_success_criteria(self, transformed_obs, action):
        success = False
        return success

    # Determines whether to terminate the current training episode.
    # This implementation always returns False, meaning episodes continue indefinitely.
    async def compute_termination(self, transformed_obs, action):
        return False