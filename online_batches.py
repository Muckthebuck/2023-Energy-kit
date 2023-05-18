from joblib import load
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
import numpy as np

# to load the preprocessed data and pass it into ml_model.py concisely
class online_batches:
    def __init__(self) -> None:
        # load pre-processed data which is split in batches (moving time window)
        self.data = {"load": load('data/cleaned_data/load_model_data.bin'),
                     "temp": load('data/cleaned_data/temp_model_data.bin'),
                     "solar": load('data/cleaned_data/solar_model_data.bin')}
    # return X for batch t
    def get_X(self, t, data_name):
        return self.data[data_name]["X_test_batch"][t]
    def get_y(self, t, data_name):
        return self.data[data_name]["Y_test_batches"][t]
    def get_timestamps(self):
        return self.data['load']['test_data_split'].index
    def get_scalar_y(self, data_name):
        return self.data[data_name]['scaler_y']
    
    def get_online_training_data(self,t):
        temp_X, temp_y = self.get_X(t,"temp"), self.get_y(t,"temp")
        solar_X, solar_y = self.get_X(t,"solar"), self.get_y(t,"solar")
        load_X, load_y = self.get_X(t,"load"), self.get_y(t,"load")

        temp_scalar = self.get_scalar_y("temp")
        solar_scalar = self.get_scalar_y("solar")
        load_scalar = self.get_scalar_y("load")
        return temp_X, temp_y, temp_scalar, solar_X, solar_y, solar_scalar, load_X, load_y, load_scalar
    