import math
import numpy as np
from composabl import Teacher

class Teacher(Teacher):
    def __init__(self, *args, **kwargs):
        # Initialize history and tracking variables.
        # obs_history: Stores the history of observed sensor data.
        # reward_history: Tracks the rewards calculated during training.
        # last_reward: Keeps a record of the most recent reward.
        # error_history: Logs errors between reference (Cref) and actual (Ca) concentrations.
        # rms_history: Maintains root mean squared (RMS) errors over time.
        # count: Tracks the number of rewards computed.
        self.obs_history = None
        self.reward_history = []
        self.last_reward = 0
        self.error_history = []
        self.rms_history = []
        self.count = 0

    # Processes sensor data if needed. By default, this function returns the data unchanged.
    # Useful for normalizing, filtering, or otherwise modifying raw sensor values.
    async def transform_sensors(self, obs, action):
        return obs

    # Modifies actions if required before applying them in the environment.
    # By default, this function passes actions through unchanged.
    async def transform_action(self, transformed_obs, action):
        return action

    # Specifies the sensors that are relevant for the task.
    # Filters the sensor list to include only essential data points.
    async def filtered_sensor_space(self):
        # Relevant sensors include temperature (T, Tc), concentration (Ca, Cref), and additional metrics.
        return ['T', 'Tc', 'Ca', 'Cref', 'Tref', 'Conc_Error', 'Eps_Yield', 'Cb_Prod']

    # Computes the reward signal based on the agent's performance.
    # Encourages minimizing the error between the reference concentration (Cref) and the actual concentration (Ca).
    async def compute_reward(self, transformed_obs, action, sim_reward):
        # Initialize observation history if this is the first step.
        if self.obs_history is None:
            self.obs_history = [transformed_obs]
            return 0.0  # No reward for the initial step.
        else:
            # Append the current observation to the history.
            self.obs_history.append(transformed_obs)

        # Calculate the squared error between the reference and actual concentrations.
        error = (float(transformed_obs['Cref']) - float(transformed_obs['Ca']))**2
        self.error_history.append(error)

        # Compute the root mean square (RMS) error over the observed history.
        rms = math.sqrt(np.mean(self.error_history))
        self.rms_history.append(rms)

        # Calculate the reward using an exponential decay function to penalize cumulative errors.
        reward = math.exp(-0.01 * np.sum(self.error_history))
        self.reward_history.append(reward)

        # Increment the reward calculation counter.
        self.count += 1
        return reward

    # Optionally restricts the agent's action space.
    # By default, this function imposes no restrictions (returns None).
    async def compute_action_mask(self, transformed_obs, action):
        return None

    # Defines the success criteria for the skill.
    # This implementation does not define any success criteria and always returns False.
    async def compute_success_criteria(self, transformed_obs, action):
        success = False
        return success

    # Determines whether to terminate the current training episode.
    # This implementation does not terminate episodes and always returns False.
    async def compute_termination(self, transformed_obs, action):
        return False