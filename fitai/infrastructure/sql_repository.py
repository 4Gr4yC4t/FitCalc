from collections.abc import Iterable
from dataclasses import replace
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import selectinload, sessionmaker

from fitai.domain.models import AiAgent, PerformanceMetric, PlayerProgress
from fitai.domain.repository import IAgentRepository
from fitai.infrastructure.data_mapper import (
    agent_from_row,
    metric_to_row,
    progress_from_row,
    progress_to_row,
)
from fitai.infrastructure.orm import (
    AgentRow,
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
            return agent_from_row(row) if row is not None else None

    def log_performance(self, metrics: Iterable[PerformanceMetric]) -> None:
        rows = [metric_to_row(metric) for metric in metrics]
        if not rows:
            return
        with self._session_factory() as session, session.begin():
            session.add_all(rows)

    def save_player_progress(
        self,
        progress: PlayerProgress,
    ) -> PlayerProgress:
        with self._session_factory() as session, session.begin():
            row = progress_to_row(progress)
            session.add(row)
            session.flush()
            return replace(progress, id=row.id, inventory=progress.inventory.copy())

    def get_player_progress(self, progress_id: int) -> PlayerProgress | None:
        with self._session_factory() as session:
            row = session.get(PlayerProgressRow, progress_id)
            return progress_from_row(row) if row is not None else None
