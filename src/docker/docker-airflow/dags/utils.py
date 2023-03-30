import yaml
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


class CosSinTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def set_output(self, transform=None):
        pass
    def transform(self, X, y=None):
        X['dow_cos'] = np.cos(2*np.pi*X['dayofweek']/6)
        X['dow_sin'] = np.sin(2*np.pi*X['dayofweek']/6)
        X['hour_cos'] = np.cos(2*np.pi*X['hour']/23)
        X['hour_sin'] = np.sin(2*np.pi*X['hour']/23)
        return X.drop(columns=['dayofweek','hour'])
    

