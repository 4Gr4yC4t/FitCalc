from collections.abc import Iterable
from datetime import datetime, timezone
import json
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import selectinload, sessionmaker

from fitai.domain.models import AiAgent, PerformanceMetric, PlayerProgress
from fitai.domain.repository import IAgentRepository
from fitai.infrastructure.orm import (
    AgentRow,
    MetricRow,
    ParameterRow,
    ParameterSetRow,
    PlayerProgressRow,
    StrategyRow,
)


class SqlAgentRepository(IAgentRepository):
    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def save_agent_state(self, agent: AiAgent) -> None:
        with self._session_factory() as session, session.begin():
            strategy = session.scalar(
                select(StrategyRow).where(
                    StrategyRow.type_name == agent.strategy
                )
            )
            if strategy is None:
                strategy = StrategyRow(type_name=agent.strategy)
                session.add(strategy)
                session.flush()

            row = session.get(AgentRow, str(agent.id))
            if row is None:
                row = AgentRow(
                    id=str(agent.id),
                    name=agent.name,
                    version=agent.version,
                    created_at=datetime.now(timezone.utc).replace(tzinfo=None),
                    strategy=strategy,
                )
                session.add(row)
            else:
                row.name = agent.name
                row.version = agent.version
                row.strategy = strategy

            parameter_set_name = f"agent:{agent.id}"
            parameter_set = session.scalar(
                select(ParameterSetRow)
                .where(ParameterSetRow.name == parameter_set_name)
                .options(selectinload(ParameterSetRow.parameters))
            )
            if parameter_set is None:
                parameter_set = ParameterSetRow(name=parameter_set_name)
                session.add(parameter_set)
            existing_parameters = {
                parameter.param_key: parameter
                for parameter in parameter_set.parameters
            }
            for key, value in agent.parameters.items():
                parameter = existing_parameters.pop(key, None)
                if parameter is None:
                    parameter_set.parameters.append(
                        ParameterRow(param_key=key, param_value=value)
                    )
                else:
                    parameter.param_value = value
            for parameter in existing_parameters.values():
                parameter_set.parameters.remove(parameter)
            if parameter_set not in row.parameter_sets:
                row.parameter_sets.append(parameter_set)

    def get_agent_by_id(self, agent_id: UUID) -> AiAgent | None:
        with self._session_factory() as session:
            row = session.scalar(
                select(AgentRow)
                .where(AgentRow.id == str(agent_id))
                .options(
                    selectinload(AgentRow.strategy),
                    selectinload(AgentRow.parameter_sets).selectinload(
                        ParameterSetRow.parameters
                    ),
                )
            )
            if row is None:
                return None

            parameters = {
                parameter.param_key: parameter.param_value
                for parameter_set in row.parameter_sets
                for parameter in parameter_set.parameters
            }
            return AiAgent(
                id=UUID(row.id),
                name=row.name,
                version=row.version,
                strategy=row.strategy.type_name,
                parameters=parameters,
            )

    def log_performance(self, metrics: Iterable[PerformanceMetric]) -> None:
        rows = [
            MetricRow(
                agent_id=str(metric.agent_id),
                timestamp=self._as_utc_naive(metric.timestamp),
                inference_ms=metric.inference_ms,
                current_fps=metric.current_fps,
                gc_allocated_mb=metric.gc_allocated_mb,
                false_positives=metric.false_positives,
            )
            for metric in metrics
        ]
        if not rows:
            return
        with self._session_factory() as session, session.begin():
            session.add_all(rows)

    def save_player_progress(
        self,
        progress: PlayerProgress,
    ) -> PlayerProgress:
        with self._session_factory() as session, session.begin():
            row = PlayerProgressRow(
                player_name=progress.player_name,
                unlocked_difficulty=progress.unlocked_difficulty,
                inventory_json=json.dumps(progress.inventory, separators=(",", ":")),
            )
            session.add(row)
            session.flush()
            return PlayerProgress(
                id=row.id,
                player_name=row.player_name,
                unlocked_difficulty=row.unlocked_difficulty,
                inventory=progress.inventory.copy(),
            )

    def get_player_progress(self, progress_id: int) -> PlayerProgress | None:
        with self._session_factory() as session:
            row = session.get(PlayerProgressRow, progress_id)
            if row is None:
                return None
            return PlayerProgress(
                id=row.id,
                player_name=row.player_name,
                unlocked_difficulty=row.unlocked_difficulty,
                inventory=json.loads(row.inventory_json),
            )

    @staticmethod
    def _as_utc_naive(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value
        return value.astimezone(timezone.utc).replace(tzinfo=None)
