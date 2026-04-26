from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

default_args = {
    "owner": "mlops-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="drone_mission_simulator",
    description="Simulates a drone returning from a patrol mission",
    schedule_interval="*/5 * * * *",   # toutes les 5 min (mode test)
    start_date=datetime(2026, 4, 1),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["drone", "simulation", "etl"],
) as dag:

    simulate_mission = BashOperator(
        task_id="simulate_mission",
        bash_command=(
            "cd /data && "
            "cp -u /opt/airflow/generate_patrol_db.py . && "
            "python generate_patrol_db.py"
        ),
    )

    # BONUS (+0.5 pt) : déclenche DAG 2 immédiatement
    trigger_sync = TriggerDagRunOperator(
        task_id="trigger_sync",
        trigger_dag_id="drone_patrol_sync",
        wait_for_completion=False,
        reset_dag_run=True,
    )