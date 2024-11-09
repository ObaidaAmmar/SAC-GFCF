import pandas as pd
import numpy as np

class GenSet(object):
    def __init__(self, name):
        self.counterfactuals = None
        self.name = name
    
    def set_counterfactuals(self, values):
        self.counterfactuals = [round(float(value), 4) if value != -999999 else value for value in values] #values
    
    def get_counterfactuals(self):
        return self.counterfactuals
    
    def delete_feature(self, idx):
        self.counterfactuals[idx] = -999999
    
    def insert_feature(self, idx, value):
        self.counterfactuals[idx] = round(float(value), 4) #value
    
    def modify_feature(self, idx, value, min_val, max_val, feature_type):
        normalized_value = 200 * ((self.counterfactuals[idx] - min_val) / (max_val - min_val)) - 100
        modified_value = np.clip(normalized_value + value, -100, 100)
        denormalized_value = (modified_value + 100) / 200 * (max_val - min_val) + min_val
        if feature_type == 'int':
            denormalized_value = round(denormalized_value)
        else:
            denormalized_value = round(denormalized_value, 4)
        self.counterfactuals[idx] = denormalized_value
        
def minMax(data):
    return pd.Series(index=['min','max'],data=[data.min(),data.max()])