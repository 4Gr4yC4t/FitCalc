"""Create the FitAI persistence schema.

Revision ID: 0001_initial_schema
Revises:
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "strategies",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("type_name", sa.String(length=32), nullable=False),
        sa.Column("description", sa.String(length=256), nullable=False),
        sa.UniqueConstraint("type_name"),
    )
    strategies = sa.table(
        "strategies",
        sa.column("type_name", sa.String(length=32)),
        sa.column("description", sa.String(length=256)),
    )
    op.bulk_insert(
        strategies,
        [
            {"type_name": "FSM", "description": "Finite-state machine strategy"},
            {"type_name": "RL", "description": "Reinforcement-learning strategy"},
            {"type_name": "Voice", "description": "Voice-command strategy"},
        ],
    )
    op.create_table(
        "ai_agents",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("version", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("current_strategy_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["current_strategy_id"], ["strategies.id"]),
    )
    op.create_table(
        "parameter_sets",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.UniqueConstraint("name"),
    )
    op.create_table(
        "parameters",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("parameter_set_id", sa.Integer(), nullable=False),
        sa.Column("param_key", sa.String(length=64), nullable=False),
        sa.Column("param_value", sa.String(length=256), nullable=False),
        sa.ForeignKeyConstraint(["parameter_set_id"], ["parameter_sets.id"]),
        sa.UniqueConstraint("parameter_set_id", "param_key"),
    )
    op.create_table(
        "agent_parameter_sets",
        sa.Column("agent_id", sa.String(length=36), primary_key=True),
        sa.Column("parameter_set_id", sa.Integer(), primary_key=True),
        sa.ForeignKeyConstraint(["agent_id"], ["ai_agents.id"]),
        sa.ForeignKeyConstraint(["parameter_set_id"], ["parameter_sets.id"]),
    )
    op.create_table(
        "performance_metrics",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("agent_id", sa.String(length=36), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("inference_ms", sa.Float(), nullable=False),
        sa.Column("current_fps", sa.Float(), nullable=False),
        sa.Column("gc_allocated_mb", sa.Float(), nullable=False),
        sa.Column("false_positives", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["agent_id"], ["ai_agents.id"]),
    )
    op.create_index(
        "ix_metrics_agent_time",
        "performance_metrics",
        ["agent_id", "timestamp"],
    )
    op.create_table(
        "player_progress",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("player_name", sa.String(length=64), nullable=False),
        sa.Column("unlocked_difficulty", sa.String(length=32), nullable=False),
        sa.Column("inventory_json", sa.Text(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("player_progress")
    op.drop_index("ix_metrics_agent_time", table_name="performance_metrics")
    op.drop_table("performance_metrics")
    op.drop_table("agent_parameter_sets")
    op.drop_table("parameters")
    op.drop_table("parameter_sets")
    op.drop_table("ai_agents")
    op.drop_table("strategies")
