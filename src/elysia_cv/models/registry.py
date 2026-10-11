"""Model plugin registry — adapted from ElysiaRobot policies/registry.py.

Discovers all model plugins via ``core/plugin.py`` discovery infrastructure
and builds an immutable name→spec mapping.
"""

from __future__ import annotations

from dataclasses import field, dataclass
from typing import Any

from elysia_cv.core.plugin import LoadedModelPlugin, discover_model_plugins
from elysia_cv.models.contract import ModelPlugin


@dataclass(frozen=True)
class ModelRegistry:
    """Immutable registry of all discovered model plugins.

    Built once at import / CLI start time; never mutated.
    """

    entries: dict[str, LoadedModelPlugin] = field(default_factory=dict)

    @classmethod
    def discover(cls) -> "ModelRegistry":
        """Scan the models/ tree and return a populated registry."""
        loaded = discover_model_plugins()
        entries: dict[str, LoadedModelPlugin] = {}
        for item in loaded:
            name = item.plugin.name
            if name in entries:
                # On conflict, first-discovered wins (built-in over community)
                continue
            entries[name] = item
        return cls(entries=entries)

    def get(self, name: str) -> LoadedModelPlugin:
        """Return a loaded plugin by name."""
        if name not in self.entries:
            available = sorted(self.entries.keys())
            raise KeyError(
                f"model plugin {name!r} not found. Available: {available}"
            )
        return self.entries[name]

    def list_models(self, task_type: str | None = None) -> list[str]:
        """List model names, optionally filtered by task_type."""
        if task_type is None:
            return sorted(self.entries.keys())
        return sorted(
            name
            for name, entry in self.entries.items()
            if getattr(entry.plugin, "task_type", None) == task_type
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, name: str) -> bool:
        return name in self.entries

    def __repr__(self) -> str:
        return f"ModelRegistry({len(self.entries)} models)"


# Singleton — lazily built
_model_registry: ModelRegistry | None = None


def get_model_registry() -> ModelRegistry:
    """Return the global (singleton) model registry."""
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry.discover()
    return _model_registry