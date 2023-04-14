from sqlalchemy import create_engine
import pandas as pd
import time

time.sleep(30)
engine = create_engine(r'postgresql+psycopg2://postgres:postgres@postgresql/postgres')
weather = pd.read_csv('weather.csv', index_col=0).to_sql('weather', engine, if_exists='replace')
cab_rides = pd.read_csv('cab_rides.csv', index_col=0).to_sql('cab_rides', engine, if_exists='replace')
