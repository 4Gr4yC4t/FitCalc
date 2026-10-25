import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from sqlalchemy import func, select, text

from fitai.domain.models import AiAgent, PerformanceMetric, PlayerProgress
from fitai.infrastructure.db import initialize_database
from fitai.infrastructure.metrics_buffer import BufferedMetricsWriter
from fitai.infrastructure.orm import AgentRow, MetricRow
from fitai.infrastructure.sql_repository import SqlAgentRepository


class PersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "fitai.sqlite3"
        self.database = initialize_database(self.db_path)
        self.repository = SqlAgentRepository(self.database.session_factory)
        self.agent = AiAgent(
            name="test-agent",
            version="1.0",
            strategy="FSM",
            parameters={"threshold": "0.75"},
            id=uuid4(),
        )

    def tearDown(self) -> None:
        self.database.close()
        self.temp_dir.cleanup()

    def test_migration_creates_schema_and_enables_foreign_keys(self) -> None:
        with self.database.engine.connect() as connection:
            tables = set(
                connection.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table'")
                ).scalars()
            )
            foreign_keys_enabled = connection.exec_driver_sql(
                "PRAGMA foreign_keys"
            ).scalar_one()

        self.assertTrue(
            {
                "strategies",
                "ai_agents",
                "parameter_sets",
                "parameters",
                "agent_parameter_sets",
                "performance_metrics",
                "player_progress",
                "alembic_version",
            }.issubset(tables)
        )
        self.assertEqual(foreign_keys_enabled, 1)

    def test_agent_upsert_round_trips_and_external_strategy_change_is_read(self) -> None:
        self.repository.save_agent_state(self.agent)
        updated_agent = AiAgent(
            id=self.agent.id,
            name="renamed-agent",
            version="2.0",
            strategy="RL",
            parameters={"threshold": "0.9", "mode": "safe"},
        )
        self.repository.save_agent_state(updated_agent)

        self.assertEqual(self.repository.get_agent_by_id(self.agent.id), updated_agent)
        with self.database.session_factory() as session:
            self.assertEqual(
                session.scalar(select(func.count()).select_from(AgentRow)),
                1,
            )
            voice_id = session.scalar(
                text("SELECT id FROM strategies WHERE type_name = 'Voice'")
            )
            session.execute(
                text(
                    "UPDATE ai_agents SET current_strategy_id = :strategy_id "
                    "WHERE id = :agent_id"
                ),
                {"strategy_id": voice_id, "agent_id": str(self.agent.id)},
            )
            session.commit()

        self.assertEqual(
            self.repository.get_agent_by_id(self.agent.id).strategy,
            "Voice",
        )

    def test_batch_metrics_preserve_timestamps_and_fields(self) -> None:
        self.repository.save_agent_state(self.agent)
        timestamp = datetime(2026, 10, 25, 12, 30, tzinfo=timezone.utc)
        metrics = [
            PerformanceMetric(
                agent_id=self.agent.id,
                timestamp=timestamp,
                inference_ms=3.5,
                current_fps=58.0,
                gc_allocated_mb=2.1,
                false_positives=1,
            ),
            PerformanceMetric(
                agent_id=self.agent.id,
                timestamp=timestamp,
                inference_ms=4.0,
                current_fps=57.0,
            ),
        ]

        self.repository.log_performance(metrics)

        with self.database.session_factory() as session:
            stored = session.scalars(
                select(MetricRow).order_by(MetricRow.id)
            ).all()
        self.assertEqual(len(stored), 2)
        self.assertEqual(stored[0].timestamp, timestamp.replace(tzinfo=None))
        self.assertEqual(stored[0].false_positives, 1)
        self.assertEqual(stored[1].gc_allocated_mb, 0.0)

    def test_player_progress_round_trips_inventory_json(self) -> None:
        saved = self.repository.save_player_progress(
            PlayerProgress(
                player_name="player-1",
                unlocked_difficulty="intermediate",
                inventory={"water": 2, "token": 5},
            )
        )

        self.assertIsNotNone(saved.id)
        self.assertEqual(
            self.repository.get_player_progress(saved.id),
            saved,
        )
        self.assertIsNone(self.repository.get_player_progress(-1))

    def test_buffer_flushes_batches_and_closes_idempotently(self) -> None:
        self.repository.save_agent_state(self.agent)
        writer = BufferedMetricsWriter(
            self.repository,
            flush_interval=30,
            max_batch=2,
        )
        for _ in range(3):
            writer.push(self._metric())

        writer.close()
        writer.close()

        with self.database.session_factory() as session:
            self.assertEqual(
                session.scalar(select(func.count()).select_from(MetricRow)),
                3,
            )
        with self.assertRaisesRegex(RuntimeError, "after the writer is closed"):
            writer.push(self._metric())

    def test_flush_drains_multiple_batches_without_waiting_for_interval(self) -> None:
        self.repository.save_agent_state(self.agent)
        writer = BufferedMetricsWriter(
            self.repository,
            flush_interval=30,
            max_batch=2,
        )
        for _ in range(3):
            writer.push(self._metric())

        writer.flush()
        writer.close()

        with self.database.session_factory() as session:
            self.assertEqual(
                session.scalar(select(func.count()).select_from(MetricRow)),
                3,
            )

    def test_buffer_surfaces_background_persistence_errors(self) -> None:
        self.repository.save_agent_state(self.agent)

        class FailingRepository:
            def log_performance(self, metrics) -> None:
                raise sqlite3.OperationalError("disk is unavailable")

        writer = BufferedMetricsWriter(
            FailingRepository(),
            flush_interval=30,
            max_batch=1,
        )
        writer.push(self._metric())
        with self.assertRaisesRegex(RuntimeError, "persistence failed") as raised:
            writer.flush()
        self.assertIsInstance(raised.exception.__cause__, sqlite3.OperationalError)
        with self.assertRaisesRegex(RuntimeError, "persistence failed"):
            writer.close()

    def test_buffer_rejects_invalid_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "flush_interval"):
            BufferedMetricsWriter(self.repository, flush_interval=0)
        with self.assertRaisesRegex(ValueError, "max_batch"):
            BufferedMetricsWriter(self.repository, max_batch=0)

    def _metric(self) -> PerformanceMetric:
        return PerformanceMetric(
            agent_id=self.agent.id,
            timestamp=datetime.now(timezone.utc),
            inference_ms=1.0,
            current_fps=60.0,
        )


if __name__ == "__main__":
    unittest.main()
