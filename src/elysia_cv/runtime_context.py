"""Cross-environment runtime context snapshot — adapted from ElysiaRobot.

RuntimeContext describes *the current machine and process environment*
only. It does NOT carry experiment semantics (those live in the training
config). Machine differences are injected through this context and
environment variables.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import sys
from typing import Any, Mapping


@dataclass(frozen=True)
class RuntimeContext:
    """Snapshot of the machine / process environment."""

    project_root: str
    cwd: str
    python_executable: str
    output_root: str | None
    hf_home: str | None
    hf_hub_cache: str | None
    offline_mode: bool
    rank: int
    local_rank: int
    world_size: int
    proc_per_node: int | None

    @classmethod
    def from_env(
        cls,
        *,
        project_root: str | Path | None = None,
        output_root: str | Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> "RuntimeContext":
        source = os.environ if env is None else env
        root = Path(project_root).resolve() if project_root is not None else _discover_project_root()
        return cls(
            project_root=str(root),
            cwd=str(Path.cwd()),
            python_executable=str(source.get("ELYSIA_PYTHON") or sys.executable),
            output_root=str(output_root) if output_root is not None else source.get("ELYSIA_OUTPUT_ROOT"),
            hf_home=_empty_to_none(source.get("ELYSIA_HF_HOME") or source.get("HF_HOME")),
            hf_hub_cache=_empty_to_none(source.get("ELYSIA_HF_HUB_CACHE") or source.get("HUGGINGFACE_HUB_CACHE")),
            offline_mode=_env_bool(source.get("HF_HUB_OFFLINE")) or _env_bool(source.get("TRANSFORMERS_OFFLINE")),
            rank=_env_int(source.get("RANK"), 0),
            local_rank=_env_int(source.get("LOCAL_RANK"), 0),
            world_size=max(_env_int(source.get("WORLD_SIZE"), 1), 1),
            proc_per_node=_optional_env_int(source.get("NPROC_PER_NODE") or source.get("PROC_PER_NODE")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _discover_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "elysia_cv").is_dir():
            return parent
    return Path.cwd().resolve()


def _empty_to_none(value: str | None) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _env_bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(value: str | None, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return int(default)


def _optional_env_int(value: str | None) -> int | None:
    text = str(value).strip() if value is not None else ""
    if not text:
        return None
    return _env_int(text, 0)