"""Conversions between the persistence schema and FitAI domain models."""

from datetime import datetime, timezone
import json
from uuid import UUID

from fitai.domain.models import AiAgent, PerformanceMetric, PlayerProgress
from fitai.infrastructure.orm import AgentRow, MetricRow, PlayerProgressRow


def agent_from_row(row: AgentRow) -> AiAgent:
    return AiAgent(
        id=UUID(row.id),
        name=row.name,
        version=row.version,
        strategy=row.strategy.type_name,
        parameters={
            parameter.param_key: parameter.param_value
            for parameter_set in row.parameter_sets
            for parameter in parameter_set.parameters
        },
    )


def metric_to_row(metric: PerformanceMetric) -> MetricRow:
    timestamp = metric.timestamp
    if timestamp.tzinfo is not None:
        timestamp = timestamp.astimezone(timezone.utc).replace(tzinfo=None)
    return MetricRow(
        agent_id=str(metric.agent_id),
        timestamp=timestamp,
        inference_ms=metric.inference_ms,
        current_fps=metric.current_fps,
        gc_allocated_mb=metric.gc_allocated_mb,
        false_positives=metric.false_positives,
    )


def progress_to_row(progress: PlayerProgress) -> PlayerProgressRow:
    return PlayerProgressRow(
        player_name=progress.player_name,
        unlocked_difficulty=progress.unlocked_difficulty,
        inventory_json=json.dumps(progress.inventory, separators=(",", ":")),
    )


def progress_from_row(row: PlayerProgressRow) -> PlayerProgress:
    return PlayerProgress(
        id=row.id,
        player_name=row.player_name,
        unlocked_difficulty=row.unlocked_difficulty,
        inventory=json.loads(row.inventory_json),
    )
