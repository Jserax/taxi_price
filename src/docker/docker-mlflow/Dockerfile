FROM python:3.8-slim
RUN pip install psycopg2-binary
RUN pip install mlflow 
RUN mkdir mlartifacts

ENTRYPOINT [ "mlflow", "server", "--host", "0.0.0.0"]
CMD [ "--backend-store-uri", "postgresql://postgres:postgres@localhost:5434/postgres", \
      "--default-artifact-root", "/mlartifacts"]