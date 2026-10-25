from datetime import datetime
import uuid

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass


class StrategyRow(Base):
    __tablename__ = "strategies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    type_name: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    description: Mapped[str] = mapped_column(String(256), default="", nullable=False)
    agents: Mapped[list["AgentRow"]] = relationship(back_populates="strategy")


class AgentRow(Base):
    __tablename__ = "ai_agents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    current_strategy_id: Mapped[int] = mapped_column(
        ForeignKey("strategies.id"), nullable=False
    )

    strategy: Mapped[StrategyRow] = relationship(back_populates="agents")
    metrics: Mapped[list["MetricRow"]] = relationship(
        back_populates="agent", cascade="all, delete-orphan"
    )
    parameter_sets: Mapped[list["ParameterSetRow"]] = relationship(
        secondary="agent_parameter_sets", back_populates="agents"
    )


class ParameterSetRow(Base):
    __tablename__ = "parameter_sets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    parameters: Mapped[list["ParameterRow"]] = relationship(
        back_populates="parameter_set", cascade="all, delete-orphan"
    )
    agents: Mapped[list[AgentRow]] = relationship(
        secondary="agent_parameter_sets", back_populates="parameter_sets"
    )


class ParameterRow(Base):
    __tablename__ = "parameters"
    __table_args__ = (
        UniqueConstraint("parameter_set_id", "param_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    parameter_set_id: Mapped[int] = mapped_column(
        ForeignKey("parameter_sets.id"), nullable=False
    )
    param_key: Mapped[str] = mapped_column(String(64), nullable=False)
    param_value: Mapped[str] = mapped_column(String(256), nullable=False)
    parameter_set: Mapped[ParameterSetRow] = relationship(
        back_populates="parameters"
    )


class AgentParameterSetRow(Base):
    __tablename__ = "agent_parameter_sets"

    agent_id: Mapped[str] = mapped_column(
        ForeignKey("ai_agents.id"), primary_key=True
    )
    parameter_set_id: Mapped[int] = mapped_column(
        ForeignKey("parameter_sets.id"), primary_key=True
    )


class MetricRow(Base):
    __tablename__ = "performance_metrics"
    __table_args__ = (Index("ix_metrics_agent_time", "agent_id", "timestamp"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    agent_id: Mapped[str] = mapped_column(ForeignKey("ai_agents.id"), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    inference_ms: Mapped[float] = mapped_column(Float, nullable=False)
    current_fps: Mapped[float] = mapped_column(Float, nullable=False)
    gc_allocated_mb: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    false_positives: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    agent: Mapped[AgentRow] = relationship(back_populates="metrics")


class PlayerProgressRow(Base):
    __tablename__ = "player_progress"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_name: Mapped[str] = mapped_column(String(64), nullable=False)
    unlocked_difficulty: Mapped[str] = mapped_column(String(32), nullable=False)
    inventory_json: Mapped[str] = mapped_column(Text, nullable=False)
