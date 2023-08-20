import datetime as dt

from airflow import DAG
from airflow.models import Variable
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.python import PythonOperator

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from numpy import pi
from airflow.operators.empty import EmptyOperator


spark_jars = '/usr/local/lib/python3.9/site-packages/pyspark/jars/postgresql-42.6.0.jar'


def load_and_process_data():
    spark = SparkSession.builder.config("spark.jars", spark_jars) \
        .master("local").appName("collect_data").getOrCreate()
    last_date = Variable.get('last_date', None)
    current_date = dt.datetime.now().timestamp()
    print(current_date)
    if last_date is None:
        last_date = dt.datetime(2000, 1, 1).timestamp()
    cab = spark.read.format("jdbc") \
        .option("url", "jdbc:postgresql://postgresql/postgres") \
        .option("driver", "org.postgresql.Driver").option("user", "postgres") \
        .option("password", "postgres") \
        .option("query", "select * from cab_rides").load()
    cab = cab.filter(cab['time_stamp'] > last_date) \
        .filter(cab['time_stamp'] <= current_date)
    weather = spark.read.format("jdbc") \
        .option("url", "jdbc:postgresql://postgresql/postgres") \
        .option("driver", "org.postgresql.Driver").option("user", "postgres") \
        .option("password", "postgres") \
        .option("query", "select * from weather").load()
    weather = weather.filter(weather['time_stamp'] > last_date) \
        .filter(weather['time_stamp'] <= current_date)
    Variable.set('last_date', current_date)
    mapping = {1: 6, 7: 5, 6: 4, 5: 3, 4: 2, 3: 1, 2: 0}
    cab = cab.withColumn("datetime", cab["time_stamp"].cast("timestamp"))
    cab = cab.withColumn('date', F.date_format(cab['datetime'], 'yyyy-MM-dd')) \
        .withColumn('hour', F.hour(cab['datetime'])) \
        .withColumn('dayofweek', F.dayofweek(cab['datetime'])) \
        .replace(mapping, subset='dayofweek')
    cab = cab.withColumn('dow_cos', F.cos(2*pi*cab['dayofweek']/6)) \
        .withColumn('dow_sin', F.sin(2*pi*cab['dayofweek']/6)) \
        .withColumn('hour_cos', F.cos(2*pi*cab['hour']/24)) \
        .withColumn('hour_sin', F.sin(2*pi*cab['hour']/24))
    for col in ['temp', 'clouds', 'pressure', 'rain', 'humidity']:
        weather = weather.withColumn(col, weather[col].cast('float'))
    weather = weather.withColumn("datetime2", weather["time_stamp"].cast("timestamp"))
    weather = weather.withColumn('date2', F.date_format(weather['datetime2'], 'yyyy-MM-dd')) \
        .withColumn('hour2', F.hour(weather['datetime2']))
    weather = weather.groupBy([weather['location'], weather['date2'], weather['hour2']]) \
        .avg('temp', 'clouds', 'pressure', 'rain', 'humidity')
    cab = cab.join(weather, (cab['date'] == weather['date2']) & \
                            (cab['source'] == weather['location']) & \
                            (cab['hour'] == weather['hour2']), how='inner') \
        .select(['destination', 'source', 'cab_type', 'name', 'distance',
                 'hour', 'dayofweek', 'avg(temp)', 'avg(clouds)', 'avg(pressure)',
                 'avg(rain)', 'avg(humidity)', 'price'])
    splits = cab.randomSplit([0.8, 0.2])
    date = dt.datetime.now().strftime('%d-%m-%y_%H:%M')
    train_set = f'/root/data/processed/train_{date}.csv'
    test_set = f'/root/data/processed/test_{date}.csv'
    Variable.set('last_train_set', train_set)
    Variable.set('last_test_set', test_set)
    splits[0].toPandas().to_csv(train_set)
    splits[1].toPandas().to_csv(test_set)


with DAG(dag_id='prepare_data',
         start_date=dt.datetime(2000, 1, 1),
         description="Data preparation for model training",
         default_args={
            "depends_on_past": False,
            "retries": 1},
         schedule_interval="*/10 * * * *",
         catchup=False,
         tags=["critical", "data"]) as dag:

    start_dag = EmptyOperator(
        task_id='start_dag')

    end_dag = EmptyOperator(
        task_id='end_dag')

    process_raw_data_task = PythonOperator(
        python_callable=load_and_process_data, task_id="load_and_process_data")

    run_training_dag = TriggerDagRunOperator(
        task_id="run_model_train", trigger_dag_id="train_model")

    start_dag >> process_raw_data_task >> run_training_dag >> end_dag
