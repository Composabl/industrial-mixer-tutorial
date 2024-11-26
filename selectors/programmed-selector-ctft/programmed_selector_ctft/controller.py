from typing import Dict, List
from composabl_core import SkillController

# The Controller class is a custom implementation of a skill controller.
# It defines logic for computing actions based on observed errors in concentration.
class Controller(SkillController):
    def __init__(self, *args, **kwargs):
        # Initialize tracking variables.
        # counter: Tracks the number of steps or iterations the controller has executed.
        self.counter = 0

    # Computes the action to be taken by the agent based on the observed concentration error.
    # If the error exceeds 10%, the controller takes corrective action.
    async def compute_action(self, obs, action):
        """
        Args:
            obs: Observations or sensor data from the environment.
            action: The current action placeholder, to be replaced with computed actions.

        Returns:
            action: The computed action based on the concentration error.
        """
        # Increment the step counter.
        self.counter += 1

        # Calculate the error between the reference concentration (Cref) and actual concentration (Ca).
        error_percentage = abs(float(obs['Ca']) - float(obs['Cref'])) / float(obs['Cref'])

        # Decide the action based on the error threshold.
        if error_percentage >= 0.1:  # If the error is 10% or more.
            action = 1  # Corrective action.
        else:
            action = 0  # No corrective action needed.

        return action

    # Transforms the observed sensor data if needed.
    # By default, this function returns the data unchanged.
    async def transform_sensors(self, obs):
        """
        Args:
            obs: Observations or sensor data from the environment.

        Returns:
            The transformed (or unchanged) sensor data.
        """
        return obs

    # Specifies the sensors that are relevant for this controller.
    # Filters the sensor space to include only essential inputs.
    async def filtered_sensor_space(self):
        """
        Returns:
            A list of sensor names relevant for the controller.
        """
        return ['T', 'Tc', 'Ca', 'Cref', 'Tref', 'Conc_Error', 'Eps_Yield', 'Cb_Prod']

    # Defines the success criteria for the skill.
    # This implementation does not define any success criteria and always returns False.
    async def compute_success_criteria(self, transformed_obs, action):
        """
        Args:
            transformed_obs: Transformed sensor data.
            action: The current action being executed.

        Returns:
            success: Boolean indicating whether the success criteria have been met.
        """
        return False

    # Determines whether to terminate the current episode.
    # This implementation does not terminate episodes and always returns False.
    async def compute_termination(self, transformed_obs, action):
        """
        Args:
            transformed_obs: Transformed sensor data.
            action: The current action being executed.

        Returns:
            terminate: Boolean indicating whether the episode should terminate.
        """
        return False