import random
from typing import Dict, List

from composabl_core import SkillController
import numpy as np
from gekko import GEKKO
from scipy import interpolate

class Controller(SkillController):
    def __init__(self, *args, **kwargs):
        self.count = 0  # Step counter to track time during simulation.
        self.display_mpc_vals = False  # Toggle for displaying MPC solution details during solve.
        remote_server = True  # Use a remote GEKKO server for computational processing.

        # Steady State Initial Conditions
        u_ss = 280.0  # Steady-state input (coolant temperature, Tc).
        Tf = 298.2  # Feed temperature (K).
        Caf = 10  # Feed concentration (kmol/m^3).

        # Initial states for concentration (Ca) and temperature (T).
        Ca_ss = 8.5698
        T_ss = 311.2612
        self.x0 = np.empty(2)
        self.x0[0] = Ca_ss  # Initial concentration.
        self.x0[1] = T_ss  # Initial temperature.

        # Initialize GEKKO for MPC.
        self.m = GEKKO(remote=remote_server)

        # Define simulation time for MPC.
        self.m.time = np.linspace(0, 90, num=90)  # 45 minutes (0.5 minute per step).

        # Initial conditions for process and control variables.
        Tc0 = 292  # Initial coolant temperature (K).
        T0 = 311  # Initial reactor temperature (K).
        Ca0 = 8.57  # Initial concentration (kmol/m^3).

        tau = self.m.Const(value=3)  # Time constant for temperature dynamics.
        Kp = self.m.Const(value=0.65)  # Process gain.

        # Manipulated Variable (Tc) and Controlled Variable (T).
        self.m.Tc = self.m.MV(value=Tc0, lb=273, ub=322)  # Manipulated variable with bounds.
        self.m.T = self.m.CV(value=T_ss)  # Controlled variable (reactor temperature).

        # Process dynamics: simple first-order dynamic model for temperature control.
        self.m.Equation(tau * self.m.T.dt() == -(self.m.T - T0) + Kp * (self.m.Tc - Tc0))

        # Configure manipulated variable (Tc).
        self.m.Tc.STATUS = 1  # Enable control of Tc.
        self.m.Tc.FSTATUS = 0  # Disable feedback status (allowing direct control).
        self.m.Tc.DMAXHI = 10  # Max upward movement per step.
        self.m.Tc.DMAXLO = -10  # Max downward movement per step.

        # Configure controlled variable (T).
        self.m.T.STATUS = 1  # Enable control feedback.
        self.m.T.FSTATUS = 1  # Use measured value for T.
        self.m.T.SP = 311  # Setpoint for reactor temperature.
        self.m.T.TR_INIT = 2  # Trajectory type.
        self.m.T.TAU = 1.0  # Time constant for trajectory tracking.

        self.m.options.CV_TYPE = 2  # Use squared error (L2 norm) as the objective.
        self.m.options.IMODE = 6  # MPC mode.
        self.m.options.SOLVER = 3  # Solver option.

        # Time interval for simulation (90 minutes total).
        time = 90
        self.t = np.linspace(0, time, time)

        # Initialize storage for simulation results.
        self.Ca = np.ones(len(self.t)) * Ca_ss  # Concentration over time.
        self.T = np.ones(len(self.t)) * T_ss  # Temperature over time.
        self.u = np.ones(len(self.t)) * u_ss  # Coolant temperature over time.

        # Define setpoints for reference transitions.
        p1 = 22  # Time to start the transition.
        p2 = 74  # Time to finish the transition.
        T_ = interpolate.interp1d([0, p1, p2, time, time + 1], [311.2612, 311.2612, 373.1311, 373.1311, 373.1311])
        C = interpolate.interp1d([0, p1, p2, time, time + 1], [8.57, 8.57, 2, 2, 2])

    # Computes the control action using the MPC solver.
    async def compute_action(self, obs, action):
        t = self.t
        i = self.count  # Current time step index.
        noise = 0  # Measurement noise level.

        # Simulate for one time period (current to next time step).
        ts = [t[i], t[i + 1]]

        # Add measurement noise.
        σ_max1 = noise * (8.5698 - 2)  # Max noise for concentration.
        σ_max2 = noise * (373.1311 - 311.2612)  # Max noise for temperature.
        σ_Ca = random.uniform(-σ_max1, σ_max1)
        σ_T = random.uniform(-σ_max2, σ_max2)
        obs['T'] += σ_T
        obs['Ca'] += σ_Ca

        # Update measurements and setpoint in the model.
        self.m.T.MEAS = obs['T']  # Measured reactor temperature.
        self.m.T.SP = obs['Tref']  # Target temperature setpoint.

        # Solve the MPC problem.
        self.m.solve(disp=self.display_mpc_vals)

        # Retrieve the new control action (coolant temperature adjustment).
        self.u[i + 1] = self.m.Tc.NEWVAL
        dTc = float(self.m.Tc.NEWVAL) - float(obs['Tc'])  # Change in coolant temperature.

        # Update initial conditions for the next time step.
        self.x0[0] = self.Ca[i + 1]
        self.x0[1] = self.T[i + 1]

        self.count += 1  # Advance the step counter.
        return [dTc]

    # Pass sensor data through unchanged (identity transformation).
    async def transform_sensors(self, obs):
        return obs

    # Select relevant sensor variables for the agent.
    async def filtered_sensor_space(self):
        return ['T', 'Tc', 'Ca', 'Cref', 'Tref', 'Conc_Error', 'Eps_Yield', 'Cb_Prod']

    # Placeholder: Defines success criteria (currently always returns False).
    async def compute_success_criteria(self, transformed_obs, action):
        return False

    # Placeholder: Defines episode termination conditions (currently always returns False).
    async def compute_termination(self, transformed_obs, action):
        return False