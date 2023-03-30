FROM python:3.8-slim
COPY requirements.txt requirements.txt
RUN apt-get update \
  && apt-get install -y \
         openjdk-11-jre \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*
COPY postgresql-42.6.0.jar /usr/local/lib/python3.8/site-packages/pyspark/jars/postgresql-42.6.0.jar
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV SPARK_HOME=/usr/local/lib/python3.8/site-packages/pyspark
ENV AIRFLOW_HOME=/root/airflow
RUN pip install --no-cache-dir apache-airflow-providers-apache-spark
RUN pip install -r requirements.txt
RUN mkdir -p /root/data/processed/
RUN mkdir -p /root/airflow/dags/
COPY /dags /root/airflow/dags/