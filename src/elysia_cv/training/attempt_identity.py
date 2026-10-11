"""Training attempt identity, recording, and atomic writes — adapted from ElysiaRobot.

Every process start MUST generate a new attempt_id.
``current_attempt.json`` is only used for parent backtracking; its ID is
never reused.

Lifecycle modes:
  fresh                      → brand-new training, no parent
  weights_only_init          → init weights from external checkpoint, no lineage inheritance
  strict_resume               → strict continuation, parent=checkpoint's attempt
  resource_changed_continuation → continuation after resource change, parent=checkpoint's attempt
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

ATTEMPT_ID_FILE = "current_attempt.json"
ATTEMPT_RECORDS_DIR = "attempts"

LifecycleMode = Literal["fresh", "weights_only_init", "strict_resume", "resource_changed_continuation"]


@dataclass(frozen=True)
class AttemptContext:
    """Single training run's attempt identity.

    Does NOT enter execution_contract, run_signature, or config hash.
    Only passed to metrics, checkpoint, and result via RunArtifacts.
    """

    attempt_id: str
    mode: LifecycleMode
    parent_attempt_id: str | None
    parent_checkpoint_dir: str | None
    parent_step: int | None
    source_checkpoint_dir: str | None = None
    source_checkpoint_hash: str | None = None
    source_attempt_id: str | None = None
    resolved_config_hash: str | None = None
    world_size: int | None = None
    global_batch_size: int | None = None

    @property
    def start_step(self) -> int:
        """Return the starting step for this attempt."""
        return int(self.parent_step or 0)

    @classmethod
    def fresh(cls, attempt_id: str) -> "AttemptContext":
        return cls(
            attempt_id=attempt_id,
            mode="fresh",
            parent_attempt_id=None,
            parent_checkpoint_dir=None,
            parent_step=None,
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "attempt_id": self.attempt_id,
            "mode": self.mode,
            "parent_attempt_id": self.parent_attempt_id,
            "parent_checkpoint_dir": self.parent_checkpoint_dir,
            "parent_step": self.parent_step,
            "start_step": self.start_step,
        }
        if self.source_checkpoint_dir is not None:
            result["source_checkpoint_dir"] = self.source_checkpoint_dir
        if self.source_checkpoint_hash is not None:
            result["source_checkpoint_hash"] = self.source_checkpoint_hash
        if self.source_attempt_id is not None:
            result["source_attempt_id"] = self.source_attempt_id
        if self.resolved_config_hash is not None:
            result["resolved_config_hash"] = self.resolved_config_hash
        if self.world_size is not None:
            result["world_size"] = self.world_size
        if self.global_batch_size is not None:
            result["global_batch_size"] = self.global_batch_size
        return result


def generate_attempt_id() -> str:
    """Generate a unique attempt ID."""
    return str(uuid.uuid4())


def create_attempt_context(
    *,
    output_dir: Path,
    mode: LifecycleMode = "fresh",
    parent_checkpoint_dir: str | None = None,
    parent_step: int | None = None,
    parent_attempt_id: str | None = None,
    source_checkpoint_dir: str | None = None,
    source_checkpoint_hash: str | None = None,
    source_attempt_id: str | None = None,
    resolved_config_hash: str | None = None,
    world_size: int | None = None,
    global_batch_size: int | None = None,
) -> AttemptContext:
    """Atomically create a new attempt.

    Lifecycle behaviour (fresh/weights_only modes never auto-inherit):

    ==================================  ========================  ================
    Mode                                parent_attempt_id         parent_checkpoint
    ==================================  ========================  ================
    fresh                               None                      None
    weights_only_init                    None (source_* fields)   no lineage inherit
    strict_resume                        checkpoint's attempt     must exist
    resource_changed_continuation        checkpoint's attempt     must exist
    ==================================  ========================  ================

    Write order:
      1. Exclusive-create ``attempts/<uuid>.json``
      2. Atomic os.replace to ``current_attempt.json`` via temp file
    """
    attempt_id = generate_attempt_id()

    # 1) Atomically write attempt record (exclusive create)
    _write_attempt_record_exclusive(
        output_dir,
        attempt_id=attempt_id,
        mode=mode,
        parent_attempt_id=parent_attempt_id,
        parent_checkpoint_dir=parent_checkpoint_dir,
        parent_step=parent_step,
        source_checkpoint_dir=source_checkpoint_dir,
        source_checkpoint_hash=source_checkpoint_hash,
        source_attempt_id=source_attempt_id,
        resolved_config_hash=resolved_config_hash,
        world_size=world_size,
        global_batch_size=global_batch_size,
    )

    # 2) Atomically write current_attempt.json
    _atomic_write_current_attempt(
        output_dir,
        attempt_id=attempt_id,
        mode=mode,
        parent_attempt_id=parent_attempt_id,
        parent_checkpoint_dir=parent_checkpoint_dir,
        parent_step=parent_step,
        source_checkpoint_dir=source_checkpoint_dir,
        source_checkpoint_hash=source_checkpoint_hash,
        source_attempt_id=source_attempt_id,
        resolved_config_hash=resolved_config_hash,
        world_size=world_size,
        global_batch_size=global_batch_size,
    )

    return AttemptContext(
        attempt_id=attempt_id,
        mode=mode,
        parent_attempt_id=parent_attempt_id,
        parent_checkpoint_dir=parent_checkpoint_dir,
        parent_step=parent_step,
        source_checkpoint_dir=source_checkpoint_dir,
        source_checkpoint_hash=source_checkpoint_hash,
        source_attempt_id=source_attempt_id,
        resolved_config_hash=resolved_config_hash,
        world_size=world_size,
        global_batch_size=global_batch_size,
    )


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _write_attempt_record_exclusive(
    output_dir: Path,
    *,
    attempt_id: str,
    mode: LifecycleMode,
    parent_attempt_id: str | None,
    parent_checkpoint_dir: str | None,
    parent_step: int | None,
    source_checkpoint_dir: str | None,
    source_checkpoint_hash: str | None = None,
    source_attempt_id: str | None = None,
    resolved_config_hash: str | None = None,
    world_size: int | None = None,
    global_batch_size: int | None = None,
) -> Path:
    """Write ``attempts/<uuid>.json`` with exclusive create guard."""
    attempts_dir = output_dir / "launch_records" / ATTEMPT_RECORDS_DIR
    attempts_dir.mkdir(parents=True, exist_ok=True)

    record: dict[str, Any] = {
        "attempt_id": attempt_id,
        "mode": mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent_attempt_id": parent_attempt_id,
        "parent_checkpoint_dir": parent_checkpoint_dir,
        "parent_step": parent_step,
        "start_step": int(parent_step or 0),
        "source_checkpoint_dir": source_checkpoint_dir,
        "source_checkpoint_hash": source_checkpoint_hash,
        "source_attempt_id": source_attempt_id,
        "resolved_config_hash": resolved_config_hash,
        "world_size": world_size,
        "global_batch_size": global_batch_size,
    }

    path = attempts_dir / f"{attempt_id}.json"
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
    except FileExistsError:
        raise FileExistsError(
            f"Attempt record already exists: {path}. "
            "Each process start MUST generate a new attempt_id."
        )
    return path


def _atomic_write_current_attempt(
    output_dir: Path,
    *,
    attempt_id: str,
    mode: LifecycleMode,
    parent_attempt_id: str | None,
    parent_checkpoint_dir: str | None,
    parent_step: int | None,
    source_checkpoint_dir: str | None,
    source_checkpoint_hash: str | None,
    source_attempt_id: str | None,
    resolved_config_hash: str | None,
    world_size: int | None,
    global_batch_size: int | None,
) -> None:
    """Atomically write ``current_attempt.json`` via temp file + os.replace."""
    records_dir = output_dir / "launch_records"
    records_dir.mkdir(parents=True, exist_ok=True)

    pid = os.getpid()
    safe_uuid = attempt_id.replace("-", "")[:12]
    tmp_name = f"current_attempt.pid{pid}.{safe_uuid}.tmp"

    payload: dict[str, Any] = {
        "attempt_id": attempt_id,
        "mode": mode,
        "parent_attempt_id": parent_attempt_id,
        "parent_checkpoint_dir": parent_checkpoint_dir,
        "parent_step": parent_step,
        "start_step": int(parent_step or 0),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if source_checkpoint_dir:
        payload["source_checkpoint_dir"] = source_checkpoint_dir
    if source_checkpoint_hash:
        payload["source_checkpoint_hash"] = source_checkpoint_hash
    if source_attempt_id:
        payload["source_attempt_id"] = source_attempt_id
    if resolved_config_hash:
        payload["resolved_config_hash"] = resolved_config_hash
    if world_size is not None:
        payload["world_size"] = world_size
    if global_batch_size is not None:
        payload["global_batch_size"] = global_batch_size

    target = records_dir / ATTEMPT_ID_FILE
    tmp = records_dir / tmp_name
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, target)
    except OSError:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def load_attempt_record(output_dir: Path, attempt_id: str) -> dict[str, Any] | None:
    """Read a single attempt history record."""
    path = output_dir / "launch_records" / ATTEMPT_RECORDS_DIR / f"{attempt_id}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def list_attempt_records(output_dir: Path) -> list[dict[str, Any]]:
    """List all attempt history records."""
    attempts_dir = output_dir / "launch_records" / ATTEMPT_RECORDS_DIR
    if not attempts_dir.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for child in sorted(attempts_dir.glob("*.json")):
        try:
            data = json.loads(child.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                records.append(data)
        except (OSError, json.JSONDecodeError):
            continue
    return records