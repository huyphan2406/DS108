"""Airflow DAG for the DS108 rainfall forecasting data pipeline.

This DAG runs all numbered scripts in src/ from _01_*.py to _08_*.py.
It also validates important outputs so the pipeline fails early when a crawl
or feature-engineering step does not produce data.
"""
from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

PROJECT_ROOT = Path(os.getenv("DS108_PROJECT_ROOT", "/opt/airflow")).resolve()
SRC_DIR = PROJECT_ROOT / "src"
FEATURE_FILE = PROJECT_ROOT / "data" / "features" / "feature_engineered_data.csv"

SCRIPT_RE = re.compile(r"^_\d{2}_.+\.py$")
SKIP_WORDS = ("old", "backup", "bak", "tmp", "test", "copy")
DATA_EXTS = (".csv", ".parquet", ".grib", ".grib2", ".nc", ".netcdf")

ERA5_SINGLE_DIRS = [
    PROJECT_ROOT / "data/raw/era5_single_level",
    PROJECT_ROOT / "data/raw/single_level",
    PROJECT_ROOT / "data/raw/single",
]

ERA5_PRESSURE_DIRS = [
    PROJECT_ROOT / "data/raw/era5_pressure_level",
    PROJECT_ROOT / "data/raw/pressure_level",
    PROJECT_ROOT / "data/raw/pressure",
    PROJECT_ROOT / "data/raw/presure",  # fallback for old typo
]


def discover_scripts() -> list[Path]:
    """Return valid numbered scripts in src/, sorted by step number."""
    if not SRC_DIR.exists():
        raise FileNotFoundError(f"Missing src directory: {SRC_DIR}")

    scripts = []
    for path in SRC_DIR.glob("_*.py"):
        name = path.name.lower()
        if SCRIPT_RE.match(path.name) and not any(word in name for word in SKIP_WORDS):
            scripts.append(path)

    scripts = sorted(scripts, key=lambda p: p.name)
    if not scripts:
        raise FileNotFoundError(f"No numbered pipeline scripts found in {SRC_DIR}")
    return scripts


def task_id(script: Path) -> str:
    return "run_" + script.stem.lstrip("_").replace("-", "_")


def prepare_directories() -> None:
    for folder in ["data/raw", "data/clean", "data/features", "outputs", "reports", "logs"]:
        path = PROJECT_ROOT / folder
        path.mkdir(parents=True, exist_ok=True)
        print(f"Prepared: {path}")


def check_project_structure() -> None:
    required = [SRC_DIR, PROJECT_ROOT / "dags", PROJECT_ROOT / "requirements.txt"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required path(s): " + ", ".join(missing))

    print("Pipeline scripts:")
    for script in discover_scripts():
        print("-", script.relative_to(PROJECT_ROOT))


def non_empty_data_files(folders: list[Path]) -> list[Path]:
    files = []
    for folder in folders:
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in DATA_EXTS and path.stat().st_size > 0:
                files.append(path)
    return sorted(files)


def validate_raw_output(label: str, folders: list[str]) -> None:
    paths = [Path(folder) for folder in folders]
    files = non_empty_data_files(paths)

    print(f"Validating {label}")
    for path in paths:
        print(f"- {path} | exists={path.exists()}")

    if not files:
        raise FileNotFoundError(
            f"No non-empty raw data files found after {label}. "
            "Check CDSAPI_KEY, network access, crawler output paths, and task logs."
        )

    print(f"OK: found {len(files)} non-empty file(s).")
    for file in files[:20]:
        print("-", file.relative_to(PROJECT_ROOT), f"({file.stat().st_size:,} bytes)")


def validate_final_feature_file() -> None:
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(f"Missing final feature file: {FEATURE_FILE}")
    if FEATURE_FILE.stat().st_size == 0:
        raise ValueError(f"Final feature file is empty: {FEATURE_FILE}")
    print(f"OK: {FEATURE_FILE} ({FEATURE_FILE.stat().st_size:,} bytes)")


def summarize_outputs() -> None:
    for folder in ["data/features", "outputs", "reports"]:
        path = PROJECT_ROOT / folder
        print(f"\n[{folder}]")
        if not path.exists():
            print("Not found")
            continue

        files = [item for item in sorted(path.rglob("*")) if item.is_file()]
        for file in files[:50]:
            print("-", file.relative_to(PROJECT_ROOT))
        if len(files) > 50:
            print(f"... truncated, total files: {len(files)}")


with DAG(
    dag_id="ds108_rainfall_pipeline",
    description="Run DS108 rainfall data pipeline from raw data to features and model outputs.",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "ds108", "retries": 0},
    tags=["ds108", "rainfall", "airflow"],
) as dag:
    check_structure = PythonOperator(
        task_id="check_project_structure",
        python_callable=check_project_structure,
    )

    prepare_dirs = PythonOperator(
        task_id="prepare_directories",
        python_callable=prepare_directories,
    )

    previous = check_structure >> prepare_dirs

    for script in discover_scripts():
        rel_script = script.relative_to(PROJECT_ROOT)
        run_step = BashOperator(
            task_id=task_id(script),
            bash_command=f"cd {PROJECT_ROOT} && python {rel_script}",
            env={
                **os.environ,
                "DS108_PROJECT_ROOT": str(PROJECT_ROOT),
                "PYTHONPATH": str(PROJECT_ROOT),
            },
        )
        previous >> run_step
        previous = run_step

        if script.name.startswith("_02_"):
            validate_single = PythonOperator(
                task_id="validate_era5_single_raw_after_crawl",
                python_callable=validate_raw_output,
                op_kwargs={
                    "label": "ERA5 single-level crawl",
                    "folders": [str(path) for path in ERA5_SINGLE_DIRS],
                },
            )
            previous >> validate_single
            previous = validate_single

        if script.name.startswith("_03_"):
            validate_pressure = PythonOperator(
                task_id="validate_era5_pressure_raw_after_crawl",
                python_callable=validate_raw_output,
                op_kwargs={
                    "label": "ERA5 pressure-level crawl",
                    "folders": [str(path) for path in ERA5_PRESSURE_DIRS],
                },
            )
            previous >> validate_pressure
            previous = validate_pressure

    validate_features = PythonOperator(
        task_id="validate_final_feature_file",
        python_callable=validate_final_feature_file,
    )

    summarize = PythonOperator(
        task_id="summarize_outputs",
        python_callable=summarize_outputs,
    )

    previous >> validate_features >> summarize
