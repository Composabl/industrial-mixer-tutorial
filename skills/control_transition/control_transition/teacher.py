import math
import numpy as np
from composabl import Teacher

class BaseCSTR(Teacher):
    def __init__(self, *args, **kwargs):
        # Initialize variables for tracking simulation data and training progress.
        # obs_history: Keeps track of all observed sensor data during the simulation.
        # reward_history: Stores all rewards calculated at each time step.
        # last_reward: Tracks the last computed reward for reference.
        # error_history: Records the errors between reference and actual concentrations over time.
        # rms_history: Logs the root mean square (RMS) error for each step.
        # count: Counts the number of times rewards have been calculated.
        self.obs_history = None
        self.reward_history = []
        self.last_reward = 0
        self.error_history = []
        self.rms_history = []
        self.last_reward = 0
        self.count = 0

    # The transform_sensors function processes raw sensor data and transforms it 
    # into a format suitable for the agent. Currently, it passes the data through unchanged.
    async def transform_sensors(self, obs, action):
        return obs

    # The transform_action function processes the agent's output action 
    # before sending it to the simulation. Currently, it passes the action unchanged.
    async def transform_action(self, transformed_obs, action):
        return action

    # The filtered_sensor_space function specifies which sensors are relevant for the agent.
    # This helps reduce unnecessary data and focuses the agent on key metrics.
    async def filtered_sensor_space(self):
        # Returns a list of sensor names relevant for the CSTR process.
        return ['T', 'Tc', 'Ca', 'Cref', 'Tref', 'Conc_Error', 'Eps_Yield', 'Cb_Prod']

    # The compute_reward function calculates the reward signal for the agent based on 
    # its current performance. It uses an exponential decay function to penalize errors.
    async def compute_reward(self, transformed_obs, action, sim_reward):
        # Initialize observation history if this is the first step.
        if self.obs_history is None:
            self.obs_history = [transformed_obs]
            return 0.0
        else:
            # Append the current observation to the history.
            self.obs_history.append(transformed_obs)

        # Compute the squared error between the reference concentration (Cref) 
        # and the actual concentration (Ca).
        error = (float(transformed_obs['Cref']) - float(transformed_obs['Ca']))**2
        self.error_history.append(error)

        # Calculate the root mean square (RMS) error based on all errors observed so far.
        rms = math.sqrt(np.mean(self.error_history))
        self.rms_history.append(rms)

        # Compute the reward using an exponential decay function of the cumulative error.
        reward = math.exp(-0.01 * np.sum(self.error_history))
        self.reward_history.append(reward)

        # Increment the count of rewards calculated.
        self.count += 1
        return reward

    # The compute_action_mask function defines restrictions on the agent's available actions.
    # This implementation does not impose any restrictions, returning None.
    async def compute_action_mask(self, transformed_obs, action):
        return None

    # The compute_success_criteria function defines the conditions for the agent's success.
    # Currently, it always returns False, meaning success criteria are not yet defined.
    async def compute_success_criteria(self, transformed_obs, action):
        success = False
        return success

    # The compute_termination function determines when the current simulation episode should end.
    # Currently, it always returns False, meaning episodes continue indefinitely.
    async def compute_termination(self, transformed_obs, action):
        return False