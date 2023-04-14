FROM python:3.8-slim

RUN pip install sqlalchemy psycopg2-binary pandas
COPY ./main.py ./main.py
COPY ./weather.csv ./weather.csv
COPY ./cab_rides.csv ./cab_rides.csv