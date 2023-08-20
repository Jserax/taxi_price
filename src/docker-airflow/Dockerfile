FROM python:3.9-slim
COPY requirements.txt requirements.txt
RUN apt-get update \
  && apt-get install -y \
         openjdk-17-jre curl iptables libdevmapper-dev\
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

COPY postgresql-42.6.0.jar /usr/local/lib/python3.9/site-packages/pyspark/jars/postgresql-42.6.0.jar
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV SPARK_HOME=/usr/local/lib/python3.9/site-packages/pyspark
ENV AIRFLOW_HOME=/root/airflow
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir -p /root/data/processed/
RUN mkdir -p /root/airflow/dags/

