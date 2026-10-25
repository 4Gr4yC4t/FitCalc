from argparse import ArgumentParser
from datetime import datetime, timezone
import math
from pathlib import Path
import time
import tracemalloc
from uuid import UUID

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sqlalchemy import select

from fitai.domain.models import AiAgent, PerformanceMetric
from fitai.infrastructure.db import initialize_database
from fitai.infrastructure.metrics_buffer import BufferedMetricsWriter
from fitai.infrastructure.orm import MetricRow
from fitai.infrastructure.sql_repository import SqlAgentRepository


DEMO_AGENT_ID = UUID("45f08a44-37e2-47a9-862f-401068bbf3d1")


def _action_for(strategy: str, iteration: int) -> str:
    if strategy == "FSM":
        return ("warmup", "train", "recover")[iteration % 3]
    if strategy == "RL":
        return ("increase load", "hold load", "reduce load", "hold load")[
            iteration % 4
        ]
    if strategy == "Voice":
        return ("start workout", "rest", "next exercise")[iteration % 3]
    raise ValueError(f"Unsupported demo strategy: {strategy}")


def _correlation(x_values: list[float], y_values: list[float]) -> float | None:
    if len(x_values) < 2 or len(x_values) != len(y_values):
        return None
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    numerator = sum(
        (x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values)
    )
    x_square_sum = sum((x - x_mean) ** 2 for x in x_values)
    y_square_sum = sum((y - y_mean) ** 2 for y in y_values)
    denominator = math.sqrt(x_square_sum * y_square_sum)
    return numerator / denominator if denominator else None


def export_metrics_plot(
    session_factory,
    agent_id: UUID,
    plot_path: str | Path,
) -> float | None:
    with session_factory() as session:
        rows = session.scalars(
            select(MetricRow)
            .where(MetricRow.agent_id == str(agent_id))
            .order_by(MetricRow.timestamp, MetricRow.id)
        ).all()

    fps = [row.current_fps for row in rows]
    allocated_mb = [row.gc_allocated_mb for row in rows]
    correlation = _correlation(allocated_mb, fps)
    figure, fps_axis = plt.subplots()
    positions = list(range(len(rows)))
    fps_axis.plot(positions, fps, label="FPS")
    fps_axis.set_ylabel("FPS")
    memory_axis = fps_axis.twinx()
    memory_axis.plot(
        positions,
        allocated_mb,
        color="tab:red",
        label="Traced memory (MB)",
    )
    memory_axis.set_ylabel("Traced memory (MB)")
    fps_axis.set_xlabel("Sample")
    figure.suptitle("Simulated FPS and traced Python memory")
    figure.tight_layout()
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)
    return correlation


def run_demo(
    *,
    db_path: str | Path | None = None,
    plot_path: str | Path = "fps_vs_gc.png",
    iterations: int = 100,
) -> tuple[str, float | None]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    database = initialize_database(db_path)
    try:
        repository = SqlAgentRepository(database.session_factory)
        agent = repository.get_agent_by_id(DEMO_AGENT_ID)
        if agent is None:
            agent = AiAgent(
                name="FitAI demo agent",
                version="1.0",
                strategy="FSM",
                parameters={"difficulty": "beginner"},
                id=DEMO_AGENT_ID,
            )
            repository.save_agent_state(agent)

        print(f"Loaded strategy from SQLite: {agent.strategy}")
        tracemalloc.start()
        try:
            with BufferedMetricsWriter(
                repository,
                flush_interval=0.5,
                max_batch=25,
            ) as writer:
                for iteration in range(iterations):
                    temporary_data = bytearray(
                        (iteration % 20 + 1) * 25_000
                    )
                    started = time.perf_counter()
                    action = _action_for(agent.strategy, iteration)
                    inference_ms = (time.perf_counter() - started) * 1000
                    allocated_bytes, _peak_bytes = tracemalloc.get_traced_memory()
                    allocated_mb = allocated_bytes / 1_000_000
                    current_fps = max(15.0, 60.0 - allocated_mb * 60.0)
                    writer.push(
                        PerformanceMetric(
                            agent_id=agent.id,
                            timestamp=datetime.now(timezone.utc),
                            inference_ms=inference_ms,
                            current_fps=current_fps,
                            gc_allocated_mb=allocated_mb,
                        )
                    )
                    if iteration < 3:
                        print(f"  iteration {iteration + 1}: {action}")
                    del temporary_data
        finally:
            tracemalloc.stop()

        correlation = export_metrics_plot(
            database.session_factory,
            agent.id,
            plot_path,
        )
        print(f"Recorded {iterations} simulated iterations.")
        print(f"Correlation (traced memory, FPS): {correlation}")
        print(f"Plot saved to: {Path(plot_path).resolve()}")
        return agent.strategy, correlation
    finally:
        database.close()


def main() -> None:
    parser = ArgumentParser(description="Run the FitAI Lab 4 persistence demo.")
    parser.add_argument(
        "--database",
        type=Path,
        help="SQLite database path (defaults to the FitCalc user-data directory)",
    )
    parser.add_argument("--plot", type=Path, default=Path("fps_vs_gc.png"))
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    run_demo(
        db_path=args.database,
        plot_path=args.plot,
        iterations=args.iterations,
    )


if __name__ == "__main__":
    main()
