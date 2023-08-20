import yaml
import pandas as pd
from sklearn.metrics import mean_squared_error


def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def bootstrap_metrics(y_test, y_pred, num_iter=500):
    df = pd.DataFrame({'test': y_test, 'pred': y_pred})
    metrics = []
    for _ in range(num_iter):
        sample = df.sample(frac=1.0, replace=True)
        metrics.append(mean_squared_error(sample['test'], sample['pred']))
    return metrics
