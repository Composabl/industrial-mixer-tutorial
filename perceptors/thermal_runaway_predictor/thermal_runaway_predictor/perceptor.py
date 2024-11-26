from composabl_core import PerceptorImpl

import os
import pickle

# Get the path to the current file to correctly locate the ML model file.
path = os.path.dirname(os.path.realpath(__file__))

# The ThermalRunawayPredict class is a custom Perceptor that uses a pre-trained ML model
# to predict thermal runaway events. The prediction is added as a new sensor variable.
class ThermalRunawayPredict(PerceptorImpl):
    def __init__(self, *args, **kwargs):
        # Initialize variables for tracking and processing.
        # y: Tracks the current prediction output from the ML model.
        # thermal_run: An optional variable to track if a thermal runaway condition has occurred (currently unused).
        # ml_model: The pre-trained machine learning model loaded from a pickle file.
        # ML_list: A list to track relevant ML-related outputs or actions (currently unused).
        # last_Tc: Stores the last observed coolant temperature (Tc) to calculate the change (ΔTc).
        self.y = 0
        self.thermal_run = 0
        self.ml_model = pickle.load(open(f"{path}/ml_models/ml_predict_temperature_122.pkl", 'rb'))
        self.ML_list = []
        self.last_Tc = 0

    # Processes sensor data and computes predictions using the ML model.
    # Outputs a new sensor variable `thermal_runaway_predict` based on the ML model's prediction.
    async def compute(self, obs_spec, obs):
        """
        Args:
            obs_spec: Specification of the observations (not used in this implementation).
            obs: Sensor data from the environment, either as a dictionary or list.

        Returns:
            A dictionary containing the prediction as a new sensor variable: `thermal_runaway_predict`.
        """
        # Convert observations to a dictionary if they are not already in that format.
        if type(obs) != dict:
            obs_keys = ['T', 'Tc', 'Ca', 'Cref', 'Tref', 'Conc_Error', 'Eps_Yield', 'Cb_Prod']
            obs = dict(zip(obs_keys, obs))  # Map predefined keys to observed values.

        # Calculate the change in coolant temperature (ΔTc).
        if self.last_Tc == 0:  # If this is the first step, initialize ΔTc with a default value.
            self.ΔTc = 5
        else:
            self.ΔTc = float(obs['Tc']) - self.last_Tc  # Compute ΔTc as the difference from the previous Tc.

        # Initialize prediction output.
        y = 0

        # Perform ML-based prediction if the current temperature exceeds a threshold.
        if float(obs['T']) >= 340:  # Threshold condition for invoking the ML model.
            # Prepare input features for the ML model.
            X = [[float(obs['Ca']), float(obs['T']), float(obs['Tc']), self.ΔTc]]

            # Use the ML model to predict the thermal runaway condition.
            y = self.ml_model.predict(X)[0]  # Predict thermal runaway (binary output: 0 or 1).

            # Optionally, check the probability output from the ML model.
            if self.ml_model.predict_proba(X)[0][1] >= 0.3:  # Confidence threshold for positive prediction.
                y = 1  # Set the prediction to 1 if the probability of runaway exceeds 30%.
                self.y = y  # Update the internal tracking variable.

        # Update the last observed coolant temperature for the next computation.
        self.last_Tc = float(obs['Tc'])

        # Return the prediction as a new sensor variable.
        return {"thermal_runaway_predict": y}

    # Defines the relevant sensors required for the Perceptor's functionality.
    # These are the sensors used as inputs for the ML model.
    def filtered_sensor_space(self, obs):
        """
        Args:
            obs: Sensor data from the environment (not used in this implementation).

        Returns:
            A list of relevant sensor names required by the Perceptor.
        """
        return ['T', 'Tc', 'Ca', 'Cref', 'Tref', 'Conc_Error', 'Eps_Yield', 'Cb_Prod']