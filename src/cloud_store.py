"""Durable feedback and response policy storage for the hosted dashboard."""

from datetime import datetime, timezone

import pandas as pd
import psycopg
from psycopg.types.json import Jsonb

from .feedback import FEEDBACK_COLUMNS
from .q_learning import reward_for


class CloudStore:
    def __init__(self, database_url):
        self.database_url = database_url

    def _connect(self):
        return psycopg.connect(
            self.database_url, connect_timeout=10, sslmode="require"
        )

    @staticmethod
    def _ensure_tables(cursor):
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analyst_feedback (
                id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                "timestamp" TIMESTAMPTZ NOT NULL,
                risk_level TEXT NOT NULL,
                threat_family TEXT NOT NULL,
                risk_score DOUBLE PRECISION NOT NULL,
                anomaly_score DOUBLE PRECISION NOT NULL,
                recommended_action TEXT NOT NULL,
                accepted BOOLEAN NOT NULL,
                "comment" TEXT NOT NULL DEFAULT ''
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS response_policy (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                q_table JSONB NOT NULL
            )
        """)

    def load_or_seed_policy(self, initial_q):
        """Seed Neon once from the bundled policy, then use Neon as the source."""
        with self._connect() as conn:
            with conn.cursor() as cursor:
                self._ensure_tables(cursor)
                cursor.execute(
                    "INSERT INTO response_policy (id, q_table) VALUES (1, %s) "
                    "ON CONFLICT (id) DO NOTHING",
                    (Jsonb(initial_q),),
                )
                cursor.execute("SELECT q_table FROM response_policy WHERE id = 1")
                return cursor.fetchone()[0]

    def load_feedback(self):
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT " + ", ".join(f'"{name}"' for name in FEEDBACK_COLUMNS) +
                    " FROM analyst_feedback ORDER BY id"
                )
                rows = cursor.fetchall()
        return pd.DataFrame(rows, columns=FEEDBACK_COLUMNS)

    def record_feedback(
        self, engine, state, action, accepted, risk_score, anomaly_score, comment
    ):
        """Save feedback and the policy update together under a row lock."""
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT q_table FROM response_policy WHERE id = 1 FOR UPDATE"
                )
                row = cursor.fetchone()
                if row is None:
                    raise RuntimeError("Response policy has not been initialized")

                engine.q = row[0]
                reward = reward_for(
                    state[0], action, "accept" if accepted else "reject"
                )
                engine.update(state, action, reward, state)

                cursor.execute(
                    """INSERT INTO analyst_feedback
                       ("timestamp", risk_level, threat_family, risk_score,
                        anomaly_score, recommended_action, accepted, "comment")
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    (
                        datetime.now(timezone.utc), state[0], state[1],
                        float(risk_score), float(anomaly_score), action,
                        bool(accepted), comment,
                    ),
                )
                cursor.execute(
                    "UPDATE response_policy SET q_table = %s WHERE id = 1",
                    (Jsonb(engine.q),),
                )
        return reward, engine.q
