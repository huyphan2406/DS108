from datetime import datetime, timedelta

from airflow.models.dag import DAG
from airflow.providers.standard.operators.bash import BashOperator


PROJECT_DIR = "/opt/airflow/project"
SCRIPTS_DIR = f"{PROJECT_DIR}/scripts"


default_args = {
    "owner": "ds108",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="ds108_rainfall_pipeline",
    description="Pipeline xây dựng dữ liệu khí tượng đa nguồn và benchmark mô hình dự báo mưa",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["ds108", "rainfall", "gsod", "era5", "lightgbm"],
) as dag:

    crawl_gsod = BashOperator(
        task_id="crawl_gsod",
        bash_command=f"cd {PROJECT_DIR} && python {SCRIPTS_DIR}/_01_crawler.py",
    )

    crawl_era5_single = BashOperator(
        task_id="crawl_era5_single",
        bash_command=f"cd {PROJECT_DIR} && python {SCRIPTS_DIR}/_02_crawls_single_level.py",
    )

    crawl_era5_pressure = BashOperator(
        task_id="crawl_era5_pressure",
        bash_command=f"cd {PROJECT_DIR} && python {SCRIPTS_DIR}/_03_crawls_pressure_level.py",
    )

    process_single_level = BashOperator(
        task_id="process_single_level",
        bash_command=f"cd {PROJECT_DIR} && python {SCRIPTS_DIR}/_04_single_level.py",
    )

    process_pressure_level = BashOperator(
        task_id="process_pressure_level",
        bash_command=f"cd {PROJECT_DIR} && python {SCRIPTS_DIR}/_05_pressure_level.py",
    )

    build_silver_dataset = BashOperator(
        task_id="build_silver_dataset",
        bash_command=f"cd {PROJECT_DIR} && python {SCRIPTS_DIR}/_06_to_silver.py",
    )

    feature_engineering = BashOperator(
        task_id="feature_engineering",
        bash_command=f"cd {PROJECT_DIR} && python {SCRIPTS_DIR}/_07_feature_engineering.py",
    )

    train_and_evaluate_model = BashOperator(
        task_id="train_and_evaluate_model",
        bash_command=f"cd {PROJECT_DIR} && python {SCRIPTS_DIR}/_08_model.py",
    )

    crawl_era5_single >> process_single_level
    crawl_era5_pressure >> process_pressure_level

    [crawl_gsod, process_single_level, process_pressure_level] >> build_silver_dataset

    build_silver_dataset >> feature_engineering >> train_and_evaluate_model