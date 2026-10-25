from collections import deque
import threading
import time

from fitai.domain.models import PerformanceMetric
from fitai.domain.repository import IAgentRepository


class BufferedMetricsWriter:
    def __init__(
        self,
        repository: IAgentRepository,
        flush_interval: float = 5.0,
        max_batch: int = 500,
    ) -> None:
        if flush_interval <= 0:
            raise ValueError("flush_interval must be positive")
        if max_batch <= 0:
            raise ValueError("max_batch must be positive")

        self._repository = repository
        self._flush_interval = flush_interval
        self._max_batch = max_batch
        self._queue: deque[PerformanceMetric] = deque()
        self._condition = threading.Condition()
        self._closing = False
        self._closed = False
        self._writing = False
        self._flush_requested = False
        self._error: Exception | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="fitai-metrics-writer",
            daemon=True,
        )
        self._thread.start()

    def push(self, metric: PerformanceMetric) -> None:
        with self._condition:
            self._raise_if_failed()
            if self._closing or self._closed:
                raise RuntimeError("Cannot push metrics after the writer is closed")
            self._queue.append(metric)
            if len(self._queue) >= self._max_batch:
                self._condition.notify_all()

    def flush(self) -> None:
        with self._condition:
            self._raise_if_failed()
            if self._closed:
                return
            self._flush_requested = True
            self._condition.notify_all()
            self._condition.wait_for(
                lambda: (not self._queue and not self._writing) or self._error is not None
            )
            self._raise_if_failed()

    def close(self) -> None:
        with self._condition:
            if not self._closed:
                self._closing = True
                self._condition.notify_all()
        self._thread.join()
        with self._condition:
            self._raise_if_failed()

    def _run(self) -> None:
        last_flush = time.monotonic()
        while True:
            with self._condition:
                while True:
                    if self._error is not None:
                        return
                    if self._queue and (
                        len(self._queue) >= self._max_batch
                        or self._flush_requested
                        or self._closing
                        or time.monotonic() - last_flush >= self._flush_interval
                    ):
                        batch = [
                            self._queue.popleft()
                            for _ in range(min(self._max_batch, len(self._queue)))
                        ]
                        self._writing = True
                        if not self._queue:
                            self._flush_requested = False
                        break
                    if self._closing and not self._queue:
                        self._closed = True
                        self._condition.notify_all()
                        return

                    remaining = self._flush_interval - (time.monotonic() - last_flush)
                    if remaining <= 0:
                        last_flush = time.monotonic()
                        continue
                    self._condition.wait(timeout=remaining)

            try:
                self._repository.log_performance(batch)
            except Exception as error:
                with self._condition:
                    for metric in reversed(batch):
                        self._queue.appendleft(metric)
                    self._error = error
                    self._writing = False
                    self._condition.notify_all()
                return

            with self._condition:
                self._writing = False
                last_flush = time.monotonic()
                self._condition.notify_all()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("Background metric persistence failed") from self._error

    def __enter__(self) -> "BufferedMetricsWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
