import datetime as dt
import numpy as np
import pandas as pd
import optuna
from utils import read_params, bootstrap_metrics
from scipy.stats import ttest_ind

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin

import mlflow
from mlflow.client import MlflowClient
from mlflow.models.signature import infer_signature
from mlflow.exceptions import MlflowException

from airflow.models import Variable
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator


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
        return X.drop(columns=['dayofweek', 'hour'])


def objective(trial, x_train, y_train, num_cols):
    alpha = trial.suggest_float("alpha", 1e-4, 1, log=True)
    max_iter = trial.suggest_int("max_iter", 200, 1000)
    penalty = trial.suggest_categorical('penalty', ['elasticnet', 'l1', 'l2'])
    if penalty == 'elasticnet':
        l1_ratio = trial.suggest_float("l1_ratio", 0, 1)
    else:
        l1_ratio = 0
    x_train = pd.get_dummies(x_train)
    cat_cols = x_train.columns[8:]
    ct = ColumnTransformer([
            ('numerical', StandardScaler(), num_cols),
            ('categorical', 'passthrough', cat_cols)])
    regressor = SGDRegressor(alpha=alpha, max_iter=max_iter,
                             penalty=penalty, l1_ratio=l1_ratio)
    pipe = Pipeline(steps=[('cossin', CosSinTransformer()),
                           ('ct', ct),
                           ('regressor', regressor)])

    mse = cross_val_score(pipe, x_train, y_train, cv=5,
                          scoring='neg_mean_squared_error').mean()
    return mse


def train_model():
    config = read_params("/root/airflow/dags/params.yaml")
    data_config = config['data']
    mlflow_config = config['mlflow']
    num_cols = data_config['num_cols']
    cat_cols = data_config['cat_cols']
    target = data_config['target']
    train_path = Variable.get('last_train_set', 'data/processed/train.csv')
    test_path = Variable.get('last_test_set', 'data/processed/test.csv')
    train = pd.read_csv(train_path, index_col=0)
    test = pd.read_csv(test_path, index_col=0)
    x_train, y_train = train.drop(columns=[target]), train[target]
    x_test, y_test = test.drop(columns=[target]), test[target]
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(study_name="taxi_price", sampler=sampler,
                                direction='maximize')
    study.optimize(lambda trial: objective(trial, x_train, y_train,
                                           num_cols), n_trials=30)

    mlflow.set_tracking_uri(mlflow_config["remote_server_uri"])
    mlflow.set_experiment(mlflow_config["experiment_name"])
    with mlflow.start_run(run_name=mlflow_config["run_name"]):
        ct = ColumnTransformer([
            ('numerical', StandardScaler(), num_cols),
            ('categorical', OneHotEncoder(sparse_output=False), cat_cols)])
        regressor = SGDRegressor(**study.best_params)
        pipe = Pipeline(steps=[('cossin', CosSinTransformer()),
                               ('ct', ct),
                               ('regressor', regressor)])
        pipe.fit(x_train, y_train)
        y_pred = pipe.predict(x_test)
        metrics = [r2_score(y_test, y_pred),
                   mean_squared_error(y_test, y_pred)]
        mlflow.log_params(study.best_params)
        mlflow.log_metric('r2_score', metrics[0])
        mlflow.log_metric('mse', metrics[1])
        signature = infer_signature(x_test, y_pred)
        input_example = x_train.iloc[[1]]
        mlflow.sklearn.log_model(pipe, "regression",
                                 registered_model_name=mlflow_config['model_name'],
                                 signature=signature,
                                 input_example=input_example)


def eval_model():
    config = read_params("/root/airflow/dags/params.yaml")
    mlflow_config = config['mlflow']
    target = config['data']['target']
    model_name = mlflow_config['model_name']
    client = MlflowClient(mlflow_config["remote_server_uri"])
    mlflow.set_tracking_uri(mlflow_config["remote_server_uri"])
    try:
        latest_ver = client.get_latest_versions(model_name)[-1].version
        current_model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")
        latest_model = mlflow.sklearn.load_model(f"models:/{model_name}/{latest_ver}")
        test_path = Variable.get('last_test_set', 'data/processed/test.csv')
        test = pd.read_csv(test_path, index_col=0)
        x_test, y_test = test.drop(columns=[target]), test[target]
        y_pred_prod = current_model.predict(x_test)
        mse_score_prod = bootstrap_metrics(y_test, y_pred_prod)
        y_pred_latest = latest_model.predict(x_test)
        mse_score_latest = bootstrap_metrics(y_test, y_pred_latest)
        test_result = ttest_ind(mse_score_prod,
                                mse_score_latest,
                                alternative="less")
        if test_result.pvalue < 0.05:
            client.transition_model_version_stage(
                name=model_name,
                version=latest_ver,
                stage="Production",
                archive_existing_versions=True)
    except MlflowException:
        client.transition_model_version_stage(
            name=model_name,
            version=1,
            stage="Production",
            archive_existing_versions=True)


with DAG(dag_id='train_model',
         start_date=dt.datetime(2000, 1, 1),
         description="Model training",
         default_args={
            "depends_on_past": False,
            "retries": 1},
         schedule_interval=None,
         catchup=False,
         tags=["critical", "train"]) as dag:

    start_dag = EmptyOperator(
        task_id='start_dag')

    end_dag = EmptyOperator(
        task_id='end_dag')

    train_model_task = PythonOperator(
        python_callable=train_model, task_id="train_model")

    eval_model_task = PythonOperator(
        python_callable=eval_model, task_id="eval_model")

    start_dag >> train_model_task >> eval_model_task >> end_dag
