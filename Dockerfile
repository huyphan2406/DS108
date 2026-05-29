FROM apache/airflow:3.2.1

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libgomp1 \
        libeccodes-dev \
        curl \
    && apt-get autoremove -yqq --purge \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

COPY requirements.txt /requirements.txt

RUN pip install --no-cache-dir \
    "apache-airflow==${AIRFLOW_VERSION}" \
    -r /requirements.txt \
    --constraint "${HOME}/constraints.txt"

COPY --chown=airflow:root dags /opt/airflow/dags
COPY --chown=airflow:root scripts /opt/airflow/project/scripts

WORKDIR /opt/airflow/project

ENV PYTHONPATH=/opt/airflow/project
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False