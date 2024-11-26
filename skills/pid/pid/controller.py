import random
from typing import Dict, List
from composabl_core import SkillController
import numpy as np

# PID Controller Function
# Implements a Proportional-Integral-Derivative (PID) controller for process control.
# Allows different numerical methods for calculating the integral and derivative actions.
def PID(TPV,
        TSP,
        Kp,
        TauI,
        TauD,
        Tkm1,
        dt=1,
        Ubias=0,
        I=0,
        N=10,
        b=1,
        c=0,
        Method='Backward'):
    """
    Args:
        TPV: The current process variable (e.g., measured temperature).
        TSP: The setpoint (desired temperature).
        Kp: Proportional gain.
        TauI: Integral time constant.
        TauD: Derivative time constant.
        Tkm1: Previous process variable.
        dt: Time step (default: 1).
        Ubias: Baseline control output.
        I: Previous integral value.
        N: Filter parameter for derivative action.
        b: Setpoint weight for proportional action.
        c: Setpoint weight for derivative action.
        Method: Numerical method for integration and differentiation.
            Options: 'Backward', 'Forward', 'Tustin', 'Ramp'.
    Returns:
        U: Control output (adjustment to manipulated variable, e.g., ΔTc).
    """
    # Calculate the error terms
    e = TSP - TPV  # Current error.
    e_before = TSP - Tkm1  # Previous error.

    # Integral and Derivative parameters
    Ti = TauI
    Td = TauD

    # Select numerical method
    if Method == 'Backward':
        b1 = Kp * dt / Ti if Ti != 0 else 0.0
        b2 = 0.0
        ad = Td / (Td + N * dt)
        bd = Kp * Td * N / (Td + N * dt)

    elif Method == 'Forward':
        b1 = 0.0
        b2 = Kp * dt / Ti if Ti != 0 else 0.0
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

    # Calculate PID components
    P = Kp * (b * e)  # Proportional term.
    I = b1 * e + b2 * e_before  # Integral term.
    D = ad * 0 + bd * ((c * e) - (c * e_before))  # Derivative term.

    # Combine components to form the control output
    U = Ubias + P + I + D

    return U


# Controller Class
# Implements a PID controller to manage the behavior of a process (e.g., reactor temperature).
class Controller(SkillController):
    def __init__(self, *args, **kwargs):
        """
        Initialize the Controller class.
        """
        self.count = 0  # Step counter.
        self.I = 0  # Initial integral value for the PID controller.
        self.T_list = []  # List to store past temperature values for derivative calculations.
        self.ΔTc = 0  # Output adjustment (change in coolant temperature).

    # Compute the control action using a PID controller
    async def compute_action(self, obs, action):
        """
        Computes the control output (adjustment to manipulated variable) based on the current observation.
        """
        # Extract target and current temperatures
        self.Tref = float(obs['Tref'])  # Setpoint (desired temperature).
        self.T = float(obs['T'])  # Current temperature.

        bias = 0  # Baseline control bias.

        # Apply PID control if not the first step
        if self.count > 0:
            self.ΔTc = PID(self.T,
                           TSP=self.Tref,
                           Kp=0.04,  # Proportional gain.
                           TauI=1.8,  # Integral time constant.
                           TauD=1.5,  # Derivative time constant.
                           Tkm1=self.T_list[-1],  # Previous temperature.
                           dt=1,  # Time step.
                           Ubias=bias,  # Control bias.
                           I=self.I,  # Integral state.
                           N=1,  # Derivative filter parameter.
                           Method='Backward')  # Numerical method.

        # Update temperature history
        self.T_list.append(self.T)

        # Increment step counter
        self.count += 1

        # Return control output as a list
        return [self.ΔTc]

    # Identity transformation for sensors
    async def transform_sensors(self, obs):
        """
        Pass sensor observations through without modification.
        """
        return obs

    # Define the relevant sensors for the process
    async def filtered_sensor_space(self):
        """
        Specify the list of relevant sensors to use for control.
        """
        return ['T', 'Tc', 'Ca', 'Cref', 'Tref', 'Conc_Error', 'Eps_Yield', 'Cb_Prod']

    # Placeholder for success criteria
    async def compute_success_criteria(self, transformed_obs, action):
        """
        Determine whether the episode has met its success criteria.
        """
        return False

    # Placeholder for termination conditions
    async def compute_termination(self, transformed_obs, action):
        """
        Determine whether to terminate the current episode.
        """
        return False