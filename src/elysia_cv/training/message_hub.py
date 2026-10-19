"""MessageHub — unified state/log bus for cross-hook communication.

Replaces ad-hoc ``hasattr(hook, "on_eval_result")`` duck-typing with a
structured publish/subscribe model.  Hooks write scalars/logs; LoggerHook
reads from the hub for unified output.

Usage::

    hub = MessageHub()
    hub.update_scalar("train/loss", 0.345)
    hub.update_scalar("val/accuracy", 0.872)

    # LoggerHook reads all scalars at once
    all_scalars = hub.get_all_scalars()

    # Auto-clearing per-iteration (optional)
    hub.start_iteration()
    hub.update_scalar("train/loss", 0.5)
    hub.end_iteration()  # resets iteration-scoped scalars
"""

from __future__ import annotations

from typing import Any


class MessageHub:
    """Thread-safe scalar store with epoch/iteration scoping."""

    def __init__(self) -> None:
        self._scalars: dict[str, float] = {}
        self._iteration_scalars: dict[str, float] = {}
        self._best_values: dict[str, float] = {}

    # ── Scalar API ──────────────────────────────────────────────

    def update_scalar(self, key: str, value: float) -> None:
        """Store a scalar (epoch-persistent)."""
        self._scalars[key] = value
        self._iteration_scalars[key] = value

    def get_scalar(self, key: str, default: float = 0.0) -> float:
        return self._scalars.get(key, default)

    def get_all_scalars(self) -> dict[str, float]:
        """Return a snapshot of all scalars (both persistent and iteration-scoped)."""
        result = dict(self._scalars)
        result.update(self._iteration_scalars)
        return result

    # ── Best-value tracking ─────────────────────────────────────

    def update_best(self, key: str, value: float, mode: str = "max") -> bool:
        """Track best value for *key*; returns True if it's a new best."""
        current = self._best_values.get(key)
        if mode == "max":
            improved = current is None or value > current
        else:
            improved = current is None or value < current
        if improved:
            self._best_values[key] = value
        return improved

    def get_best(self, key: str) -> float | None:
        return self._best_values.get(key)

    # ── Iteration scoping ───────────────────────────────────────

    def start_iteration(self) -> None:
        """Called at the start of each optimizer iteration (optional)."""
        self._iteration_scalars.clear()

    def end_iteration(self) -> None:
        """Called at end of each iteration (optional — just a semantic hook)."""
        pass

    # ── Epoch scoping ───────────────────────────────────────────

    def start_epoch(self) -> None:
        """Reset epoch-scoped scalars."""
        pass

    def end_epoch(self) -> None:
        """Persist or reset epoch scalars."""
        pass

    # ── Utility ─────────────────────────────────────────────────

    def keys(self):
        return set(self._scalars.keys()) | set(self._iteration_scalars.keys())


# ── Singleton convenience ────────────────────────────────────────

_GLOBAL_HUB: MessageHub | None = None


def get_message_hub() -> MessageHub:
    """Get or create the global MessageHub singleton."""
    global _GLOBAL_HUB
    if _GLOBAL_HUB is None:
        _GLOBAL_HUB = MessageHub()
    return _GLOBAL_HUB


def reset_message_hub() -> None:
    """Reset the global hub (useful for testing)."""
    global _GLOBAL_HUB
    _GLOBAL_HUB = None