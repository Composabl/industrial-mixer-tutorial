# Copyright (C) Composabl, Inc - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

from composabl_core import PerceptorImpl

#######
import os
import pickle

path = os.path.dirname(os.path.realpath(__file__))

class ThermalRunawayPredict(PerceptorImpl):
    def __init__(self, *args, **kwargs):
        self.y = 0
        self.thermal_run = 0
        self.ml_model = pickle.load(open(f"{path}/ml_models/ml_predict_temperature_122.pkl", 'rb'))
        self.ML_list = []
        self.last_Tc = 0

    async def compute(self, obs_spec, obs):
        # change obs to dictionary using sensors
        if type(obs) != dict:
            obs_keys = ['T', 'Tc', 'Ca', 'Cref', 'Tref','Conc_Error', 'Eps_Yield', 'Cb_Prod']
            obs = dict(zip(obs_keys, obs))

        #get the action - add action to perception
        #self.ΔTc = action[0]
        if self.last_Tc == 0:
            self.ΔTc = 5
        else:
            self.ΔTc = float(obs['Tc']) - self.last_Tc

        y = 0

        if float(obs['T']) >= 340:
            X = [[float(obs['Ca']), float(obs['T']), float(obs['Tc']), self.ΔTc]]
            y = self.ml_model.predict(X)[0]
            #print(self.ml_model.predict_proba(X))
            if self.ml_model.predict_proba(X)[0][1] >= 0.3:
                y = 1
                self.y = y

        self.last_Tc = float(obs['Tc'])

        return {"thermal_runaway_predict": y}

    def filtered_sensor_space(self, obs):
        return ['T', 'Tc', 'Ca', 'Cref', 'Tref','Conc_Error', 'Eps_Yield', 'Cb_Prod']

