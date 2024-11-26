from typing import Dict, List
from composabl_core import SkillController

# The Controller class is a custom implementation of a skill controller.
# It defines logic for computing actions based on observations and tracks state during execution.
class Controller(SkillController):
    def __init__(self, *args, **kwargs):
        # Initialize tracking variables.
        # counter: Keeps track of the number of steps or iterations since the controller started.
        self.counter = 0

    # Computes the action to be taken by the agent based on the current step.
    # Implements a simple time-based logic for deciding actions.
    async def compute_action(self, obs, action):
        """
        Args:
            obs: Observations or sensor data from the environment.
            action: The current action placeholder, to be replaced with computed actions.

        Returns:
            action: The computed action based on the step counter.
        """
        # Increment the step counter.
        self.counter += 1

        # Define action logic based on the current step range.
        if self.counter >= 0 and self.counter <= 22:
            action = 0  # Action 0 for the first 22 steps.
        elif self.counter >= 76:
            action = 2  # Action 2 after step 76.
        else:
            action = 1  # Default action for steps between 23 and 75.

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