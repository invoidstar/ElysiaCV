"""Cadence controller — manages log / checkpoint / eval firing intervals.

Keeps timing logic separate from hooks, so hooks only contain "what to do
when fired", not "when to fire".
"""

from __future__ import annotations


class CadenceController:
    """Decides which events fire at a given step."""

    def __init__(
        self,
        log_freq: int = 50,
        checkpoint_freq: int = 5000,
        eval_freq: int = 5000,
    ) -> None:
        self.log_freq = log_freq
        self.checkpoint_freq = checkpoint_freq
        self.eval_freq = eval_freq

    @staticmethod
    def _should(step: int, freq: int) -> bool:
        return freq > 0 and step > 0 and step % freq == 0

    @property
    def should_log(self) -> bool:
        return self._should(self._step, self.log_freq)

    @property
    def should_checkpoint(self) -> bool:
        return self._should(self._step, self.checkpoint_freq)

    @property
    def should_eval(self) -> bool:
        return self._should(self._step, self.eval_freq)

    def set_step(self, step: int) -> None:
        self._step = step