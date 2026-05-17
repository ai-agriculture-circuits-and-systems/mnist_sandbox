"""Early stopping when validation metrics stop improving significantly."""

from __future__ import annotations


class EarlyStopping:
    """Stop training when a monitored metric fails to improve for ``patience`` epochs."""

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.1,
        mode: str = "max",
    ) -> None:
        """
        Args:
            patience: Epochs without significant improvement before stopping.
            min_delta: Minimum change to count as improvement.
            mode: ``max`` for accuracy-like metrics, ``min`` for loss-like metrics.
        """
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_metric: float | None = None
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.stopped_epoch = 0
        self.should_stop = False

    def _is_improvement(self, metric: float) -> bool:
        if self.best_metric is None:
            return True
        if self.mode == "max":
            return metric > self.best_metric + self.min_delta
        return metric < self.best_metric - self.min_delta

    def step(self, metric: float, epoch: int) -> bool:
        """
        Update state with the latest metric.

        Returns:
            True if training should stop.
        """
        if self._is_improvement(metric):
            self.best_metric = metric
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.patience:
            self.should_stop = True
            self.stopped_epoch = epoch
            return True
        return False
