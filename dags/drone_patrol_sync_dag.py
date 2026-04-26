"""
DAG 2 — Extract → Transform → Load.
Récupère les détections non traitées du drone, filtre confiance >= 0.65,
les charge dans app_detections.db, puis les marque processed=1.
"""
import sqlite3
from datetime import datetime

from airflow import DAG
from airflow.decorators import task

DRONE_DB = "/data/drone_patrol.db"
APP_DB = "/data/app_detections.db"
CONFIANCE_THRESHOLD = 0.65

with DAG(
    dag_id="drone_patrol_sync",
    description="ETL : extract → transform → load drone detections into app DB",
    schedule_interval=None,  # déclenché par DAG 1 (bonus chaînage)
    start_date=datetime(2026, 4, 1),
    catchup=False,
    tags=["drone", "etl"],
) as dag:

    @task
    def extract() -> list:
        """Lit toutes les lignes avec processed=0."""
        conn = sqlite3.connect(DRONE_DB)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM drone_detections WHERE processed = 0"
        ).fetchall()
        conn.close()
        data = [dict(r) for r in rows]
        print(f"[extract] {len(data)} détections non traitées récupérées")
        return data

    @task
    def transform(rows: list) -> dict:
        """Filtre confiance >= 0.65 et garde tous les IDs pour marquer processed=1."""
        kept = [r for r in rows if r["confiance"] >= CONFIANCE_THRESHOLD]
        all_ids = [r["id"] for r in rows]
        print(f"[transform] {len(kept)}/{len(rows)} détections au-dessus du seuil {CONFIANCE_THRESHOLD}")
        return {"kept": kept, "all_ids": all_ids}

    @task
    def load(payload: dict) -> int:
        """Insère dans app_detections.db, puis marque processed=1 dans drone_patrol.db."""
        kept = payload["kept"]
        all_ids = payload["all_ids"]

        # 1. Crée la table si absente + insère
        app = sqlite3.connect(APP_DB)
        app.execute("""
            CREATE TABLE IF NOT EXISTS app_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                confiance REAL NOT NULL,
                model_name TEXT NOT NULL,
                source TEXT NOT NULL,
                drone_id TEXT
            )
        """)
        for r in kept:
            app.execute(
                """INSERT INTO app_detections
                   (timestamp, latitude, longitude, confiance, model_name, source, drone_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    r["timestamp"],
                    r["latitude"],
                    r["longitude"],
                    r["confiance"],
                    "drone_patrol_pipeline",
                    "drone_patrol",
                    r["drone_id"],
                ),
            )
        app.commit()
        app.close()
        print(f"[load] {len(kept)} détections insérées dans app_detections.db")

        # 2. Marque TOUTES les lignes extraites comme processed=1
        # (sinon les filtrées seraient re-récupérées indéfiniment)
        if all_ids:
            drone = sqlite3.connect(DRONE_DB)
            drone.executemany(
                "UPDATE drone_detections SET processed = 1 WHERE id = ?",
                [(i,) for i in all_ids],
            )
            drone.commit()
            drone.close()
            print(f"[load] {len(all_ids)} lignes marquées processed=1 dans drone_patrol.db")

        return len(kept)

    raw = extract()
    payload = transform(raw)
    load(payload)