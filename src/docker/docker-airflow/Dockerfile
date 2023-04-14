FROM python:3.8-slim
COPY requirements.txt requirements.txt
RUN apt-get update \
  && apt-get install -y \
         openjdk-11-jre curl iptables libdevmapper-dev\
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

COPY postgresql-42.6.0.jar /usr/local/lib/python3.8/site-packages/pyspark/jars/postgresql-42.6.0.jar
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV SPARK_HOME=/usr/local/lib/python3.8/site-packages/pyspark
ENV AIRFLOW_HOME=/root/airflow
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir -p /root/data/processed/
RUN mkdir -p /root/airflow/dags/
RUN mkdir mlartifacts
RUN mkdir dockers
WORKDIR dockers
RUN curl https://download.docker.com/linux/ubuntu/dists/focal/pool/stable/amd64/containerd.io_1.6.20-1_amd64.deb --output containerd.deb
RUN curl https://download.docker.com/linux/ubuntu/dists/focal/pool/stable/amd64/docker-ce_23.0.1-1~ubuntu.20.04~focal_amd64.deb --output docker-ce.deb
RUN curl https://download.docker.com/linux/ubuntu/dists/focal/pool/stable/amd64/docker-ce-cli_23.0.1-1~ubuntu.20.04~focal_amd64.deb --output docker-ce-cli.deb
RUN curl https://download.docker.com/linux/ubuntu/dists/focal/pool/stable/amd64/docker-buildx-plugin_0.10.4-1~ubuntu.20.04~focal_amd64.deb --output docker-buildx.deb
RUN dpkg -i containerd.deb docker-ce.deb docker-ce-cli.deb docker-buildx.deb

