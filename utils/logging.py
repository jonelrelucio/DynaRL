import csv
import os
from abc import ABC, abstractmethod


class Logger(ABC):
    """Abstract base for training metric loggers.

    Implement this to add a new backend (TensorBoard, W&B, etc.) without
    touching the Trainer or training loop.
    """

    @abstractmethod
    def log(self, episode: int, reward: float,
            epsilon: float, avg_td_error: float) -> None:
        """Record one episode's metrics."""
        ...

    def close(self) -> None:
        """Release any held resources (files, connections).
        Called once at the end of training. Safe to override or leave as-is.
        """


class CSVLogger(Logger):
    """Appends one row per episode to a CSV file.

    Resumes an existing file gracefully (no duplicate header).
    """

    def __init__(self, path: str):
        existed = os.path.exists(path)
        self._f = open(path, "a", newline="")
        self._writer = csv.writer(self._f)
        if not existed:
            self._writer.writerow(["episode", "reward", "epsilon", "avg_td_error"])

    def log(self, episode: int, reward: float,
            epsilon: float, avg_td_error: float) -> None:
        self._writer.writerow(
            [episode, f"{reward:.4f}", f"{epsilon:.6f}", f"{avg_td_error:.6f}"]
        )
        self._f.flush()

    def close(self) -> None:
        self._f.close()


class ConsoleLogger(Logger):
    """Prints a summary line every `interval` episodes."""

    def __init__(self, interval: int = 25):
        self.interval = interval
        self._rewards: list = []

    def log(self, episode: int, reward: float,
            epsilon: float, avg_td_error: float) -> None:
        self._rewards.append(reward)
        if episode % self.interval == 0:
            window = self._rewards[-self.interval:]
            avg = sum(window) / len(window)
            print(
                f"\n  ├ avg_reward {avg:8.2f}"
                f"  TD_err {avg_td_error:.4f}"
            )


class CompositeLogger(Logger):
    """Fan-out logger: delegates to multiple backends simultaneously.

    Example::

        logger = CompositeLogger(CSVLogger("run.csv"), TensorBoardLogger(...))
    """

    def __init__(self, *loggers: Logger):
        self._loggers = loggers

    def log(self, episode: int, reward: float,
            epsilon: float, avg_td_error: float) -> None:
        for lg in self._loggers:
            lg.log(episode, reward, epsilon, avg_td_error)

    def close(self) -> None:
        for lg in self._loggers:
            lg.close()
