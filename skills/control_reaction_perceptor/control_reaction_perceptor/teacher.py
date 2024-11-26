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
        # count: Counts the number of reward computations.
        self.obs_history = None
        self.reward_history = []
        self.last_reward = 0
        self.error_history = []
        self.rms_history = []
        self.last_reward = 0
        self.count = 0

    # Transforms sensor data if needed. Currently, it passes the observation through unmodified.
    # This method can be extended for normalization or unit conversion.
    async def transform_sensors(self, obs, action):
        return obs

    # Transforms the action based on observations.
    # This includes adjusting the action (coolant temperature change, ΔTc) in case of a potential thermal runaway scenario.
    async def transform_action(self, transformed_obs, action):
        # Extract the coolant temperature change from the action array.
        self.ΔTc = action[0]

        # Handle different observation formats (dictionary or list-like).
        if type(transformed_obs) == dict:
            # If the observation is a dictionary, extract the key 'observation' or a specific prediction field.
            if 'observation' in list(transformed_obs.keys()):
                transformed_obs = transformed_obs['observation']
                y = transformed_obs[5]  # Extract thermal runaway prediction or other key data.
            else:
                y = transformed_obs['thermal_runaway_predict']
        else:
            y = transformed_obs[5]  # For list-like observations, use index 5.

        # Apply machine learning-based constraints to avoid thermal runaway.
        if y == 1:  # If a thermal runaway is predicted:
            # Reduce the magnitude of ΔTc to slow down the cooling or heating process.
            self.ΔTc -= 0.05 * abs(self.ΔTc) * np.sign(self.ΔTc)

        # Reconstruct the action array with the adjusted ΔTc.
        action = np.array([self.ΔTc])
        return action

    # Filters the sensor space to include only relevant sensors for the skill.
    async def filtered_sensor_space(self):
        # Select relevant sensors such as temperature (T, Tc), concentration (Ca, Cref), and other metrics.
        return ['T', 'Tc', 'Ca', 'Cref', 'Tref', 'Conc_Error', 'Eps_Yield', 'Cb_Prod']

    # Computes the reward for the agent's actions based on observed sensor data.
    # Rewards are based on minimizing the error between the reference concentration (Cref) and actual concentration (Ca).
    async def compute_reward(self, transformed_obs, action, sim_reward):
        # Initialize observation history if it's the first call.
        if self.obs_history is None:
            self.obs_history = [transformed_obs]
            return 0.0
        else:
            # Append the current observation to the history.
            self.obs_history.append(transformed_obs)

        # Calculate the squared error between Cref and Ca.
        error = (float(transformed_obs['Cref']) - float(transformed_obs['Ca']))**2
        self.error_history.append(error)

        # Compute the root mean square (RMS) error over all recorded errors.
        rms = math.sqrt(np.mean(self.error_history))
        self.rms_history.append(rms)

        # Calculate the reward using an exponential decay function of the cumulative error.
        reward = math.exp(-0.01 * np.sum(self.error_history))
        self.reward_history.append(reward)

        # Increment the count of reward computations.
        self.count += 1
        return reward

    # Optionally restrict the set of actions available to the agent.
    # Returns None, indicating no restrictions in this implementation.
    async def compute_action_mask(self, transformed_obs, action):
        return None

    # Defines the success criteria for the skill.
    # Currently, it always returns False, indicating no success condition is set yet.
    async def compute_success_criteria(self, transformed_obs, action):
        success = False
        return success

    # Determines whether to terminate the current training episode.
    # Always returns False, meaning episodes continue indefinitely.
    async def compute_termination(self, transformed_obs, action):
        return False