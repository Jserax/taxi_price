from sqlalchemy import create_engine
import datetime as dt
import pandas as pd


def push_data():
    now = dt.datetime.now()
    ts = {1680159600: (now + dt.timedelta(minutes=0)).timestamp(),
          1680163200: (now + dt.timedelta(minutes=10)).timestamp(),
          1680166800: (now + dt.timedelta(minutes=20)).timestamp(),
          1680170400: (now + dt.timedelta(minutes=30)).timestamp(),
          1680174000: (now + dt.timedelta(minutes=40)).timestamp()}
    engine = create_engine(
        r'postgresql+psycopg2://postgres:postgres@postgresql/postgres'
        )
    weather = pd.read_csv('weather.csv', index_col=0)
    weather['time_stamp'] = weather['time_stamp'].map(ts)
    weather.to_sql('weather', engine, if_exists='replace')
    cab_rides = pd.read_csv('cab_rides.csv', index_col=0)
    cab_rides['time_stamp'] = cab_rides['time_stamp'].map(ts)
    cab_rides.to_sql('cab_rides', engine, if_exists='replace')


if __name__ == "__main__":
    push_data()
