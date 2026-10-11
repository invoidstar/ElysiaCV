"""Configuration loader — YAML file ingestion with _base_ inheritance,
override merging, and fail-fast validation.

Design
------
* **``_base_`` inheritance**: a config YAML may list a ``_base_`` key (single
  path or list of paths) referencing ancestor configs.  Loading resolves
  them from the innermost child outward, so child keys override parent keys.
  This is the standard pattern from MMDetection / MMSegmentation.

* **Override system** (adapted from ElysiaRobot's ``native_loader.py``):
  ``--set key=value`` overrides are parsed with type coercion (null / bool /
  int / float / json / string) and merged into the final config via dotted
  path assignment (e.g. ``model.pretrained_path=/tmp/ckpt.pth``).

* **Environment expansion**: ``${VAR}`` and ``${VAR:-default}`` patterns in
  string values are expanded recursively (from ElysiaRobot's
  ``configs/environment.py``).

* **Strict key validation**: unknown keys anywhere in the config tree are
  rejected immediately, with did-you-mean suggestions via ``difflib``.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

# ---------------------------------------------------------------------------
# Known section and field definitions for fail-fast validation
# ---------------------------------------------------------------------------

_CV_TOP_LEVEL_SECTIONS = frozenset(
    {"_base_", "run", "model", "data", "engine", "optim", "schedule", "tracking", "eval"}
)

_CV_FIELD_MAP: dict[tuple[str, ...], frozenset[str]] = {
    (): _CV_TOP_LEVEL_SECTIONS,
    ("run",): frozenset({"name", "stage", "seed", "output_dir", "test_run"}),
    ("model",): frozenset({"type", "pretrained_path", "backbone", "neck", "head", "kwargs"}),
    ("model", "backbone"): frozenset({"type"}),
    ("model", "neck"): frozenset({"type"}),
    ("model", "head"): frozenset({"type"}),
    ("data",): frozenset({"train", "val", "test"}),
    ("data", "train"): frozenset({"dataset_name", "dataset_root", "split", "loader", "transforms"}),
    ("data", "val"): frozenset({"dataset_name", "dataset_root", "split", "loader", "transforms"}),
    ("data", "test"): frozenset({"dataset_name", "dataset_root", "split", "loader", "transforms"}),
    ("data", "train", "loader"): frozenset(
        {"batch_size", "num_workers", "prefetch_factor", "pin_memory", "persistent_workers", "drop_last"}
    ),
    ("data", "val", "loader"): frozenset(
        {"batch_size", "num_workers", "prefetch_factor", "pin_memory", "persistent_workers", "drop_last"}
    ),
    ("data", "test", "loader"): frozenset(
        {"batch_size", "num_workers", "prefetch_factor", "pin_memory", "persistent_workers", "drop_last"}
    ),
    ("engine",): frozenset({"backend", "strategy", "distributed", "precision", "compile", "optimization"}),
    ("engine", "distributed"): frozenset(
        {
            "proc_per_node", "node_count", "master_addr", "master_port",
            "ddp_find_unused_parameters", "ddp_broadcast_buffers", "ddp_bucket_cap_mb",
            "ddp_gradient_as_bucket_view", "ddp_static_graph", "fsdp_enable",
        }
    ),
    ("engine", "precision"): frozenset({"mixed_precision", "dtype", "cudnn_benchmark", "allow_tf32"}),
    ("engine", "compile"): frozenset({"enabled", "mode"}),
    ("engine", "optimization"): frozenset({"gradient_accumulation_steps", "grad_clip_norm", "optimizer_zero_grad_set_to_none"}),
    ("optim",): frozenset({"type", "params"}),
    ("schedule",): frozenset({"type", "steps", "warmup_steps", "base_lr", "min_lr", "params"}),
    ("tracking",): frozenset({"log_freq", "wandb"}),
    ("tracking", "wandb"): frozenset({"enabled", "project", "entity", "run_id", "resume", "mode"}),
    ("eval",): frozenset({"eval_freq", "metrics", "save_best", "save_final"}),
}

# Prefixes that allow arbitrary sub-keys (plugin-owned): these paths are NOT
# validated for known keys and pass through to the model / data / optim builders.
_OPEN_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("model", "backbone"),
    ("model", "neck"),
    ("model", "head"),
    ("model", "kwargs"),
    ("optim", "params"),
    ("schedule", "params"),
)

_ENV_EXPR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_cv_config(
    path: str | Path,
    overrides: Sequence[str] | Mapping[str, Any] = (),
) -> dict[str, Any]:
    """Load a CV config YAML and return the resolved, validated dict.

    Parameters
    ----------
    path :
        Path to the root config YAML file.
    overrides :
        Either a sequence of ``"key=value"`` strings or a mapping.  Keys use
        dotted paths, e.g. ``"model.pretrained_path=/tmp/ckpt"``,
        ``"schedule.base_lr=0.01"``.

    Returns
    -------
    dict
        Resolved config dict (suitable for CVConfig construction).
    """
    raw = _load_yaml_with_base(Path(path))
    parsed_overrides = _parse_overrides(overrides)
    merged = _deep_merge(raw, parsed_overrides)
    _validate_config_mapping(merged)
    return merged


def load_cv_config_with_object(
    path: str | Path,
    overrides: Sequence[str] | Mapping[str, Any] = (),
):
    """Like :func:`load_cv_config`, but returns a typed :class:`CVConfig` object."""
    from elysia_cv.config.config import CVConfig

    raw = load_cv_config(path, overrides)
    config = CVConfig.from_mapping(raw)
    config.validate()
    return config


# ---------------------------------------------------------------------------
# YAML loading with _base_ inheritance
# ---------------------------------------------------------------------------


def _load_yaml_with_base(path: Path) -> dict[str, Any]:
    """Load a YAML file and resolve ``_base_`` references recursively.

    Resolution is depth-first: children have higher priority than parents.
    """
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError as exc:
        raise ValueError("YAML config loading requires PyYAML") from exc
    loaded = yaml.safe_load(text) or {}
    if not isinstance(loaded, Mapping):
        raise ValueError(f"config file {path!s} must contain a mapping")

    loaded = dict(loaded)
    base_refs = loaded.pop("_base_", None)

    if base_refs is None:
        return expand_environment_values(loaded)

    # Normalize to list
    if isinstance(base_refs, str):
        base_refs = [base_refs]
    if not isinstance(base_refs, list):
        raise ValueError(f"_base_ must be a string or list of strings, got {type(base_refs)}")

    # Resolve relative to the child config directory
    config_dir = path.parent

    # Load base configs left to right, then apply child on top
    merged: dict[str, Any] = {}
    for ref in base_refs:
        ref_path = (config_dir / ref).resolve()
        if not ref_path.is_file():
            raise FileNotFoundError(f"_base_ reference not found: {ref_path}")
        merged = _deep_merge(merged, _load_yaml_with_base(ref_path))

    merged = _deep_merge(merged, loaded)
    return expand_environment_values(merged)


# ---------------------------------------------------------------------------
# Environment variable expansion (from ElysiaRobot configs/environment.py)
# ---------------------------------------------------------------------------


def expand_environment_values(value: Any) -> Any:
    """Recursively expand ``${VAR}`` and ``${VAR:-default}`` patterns."""
    if isinstance(value, str):
        return _ENV_EXPR_RE.sub(_replace_environment_match, value)
    if isinstance(value, Mapping):
        return {key: expand_environment_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_environment_values(item) for item in value]
    return value


def _replace_environment_match(match: re.Match[str]) -> str:
    name = match.group(1)
    fallback = match.group(2)
    if name in os.environ:
        return os.environ[name]
    if fallback is not None:
        return fallback
    return match.group(0)


# ---------------------------------------------------------------------------
# Override parsing (adapted from ElysiaRobot native_loader.py)
# ---------------------------------------------------------------------------


def _parse_overrides(
    overrides: Sequence[str] | Mapping[str, Any],
) -> dict[str, Any]:
    """Parse CLI overrides into a nested dict.

    Accepts either ``"key=value"`` strings or a mapping.
    Type coercion: null → None, true/false → bool, integer → int,
    float → float, JSON objects/arrays → parsed, commma-sep → list.
    """
    if isinstance(overrides, Mapping):
        parsed: dict[str, Any] = {}
        for key, value in overrides.items():
            key_text = str(key).strip()
            if not key_text:
                raise ValueError("override key cannot be empty")
            if key_text in _CV_TOP_LEVEL_SECTIONS - {"_base_"} and isinstance(value, Mapping):
                parsed[key_text] = _deep_merge(parsed.get(key_text, {}), value)
                continue
            _assign_path(parsed, _validate_override_path(key_text), value)
        return parsed

    parsed = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("override key cannot be empty")
        _assign_path(parsed, _validate_override_path(key), _parse_value(raw.strip()))
    return parsed


def _validate_override_path(key: str) -> list[str]:
    """Validate a dotted override path; raise on unknown keys."""
    parts = [part for part in key.split(".") if part]
    if len(parts) < 1:
        raise ValueError(f"empty override key")
    if parts[0] not in _CV_TOP_LEVEL_SECTIONS:
        raise ValueError(f"unknown override key: {key!r}")

    tuple_parts = tuple(parts)
    # Open prefixes allow arbitrary sub-keys
    if any(tuple_parts[:len(prefix)] == prefix for prefix in _OPEN_PREFIXES):
        return parts

    for depth in range(1, len(parts)):
        prefix = tuple(parts[:depth])
        allowed = _CV_FIELD_MAP.get(prefix)
        if allowed is not None and parts[depth] not in allowed:
            suggestion = _closest_key(parts[depth], allowed)
            suffix = f"; did you mean {suggestion!r}?" if suggestion else ""
            raise ValueError(f"unknown override key: {key!r}{suffix}")
    return parts


def _parse_value(value: str) -> Any:
    """Type coercion for override values.

    Order: null → int → float → JSON → comma-list → string.
    """
    stripped = value.strip()
    lowered = stripped.lower()
    if lowered in {"null", "none"}:
        return None
    if stripped.startswith(("{", "[")):
        return json.loads(stripped)
    if lowered in {"true", "false"}:
        return lowered == "true"
    if "," in stripped:
        return [part.strip() for part in stripped.split(",") if part.strip()]
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        return stripped


def _assign_path(target: dict[str, Any], path: list[str], value: Any) -> None:
    """Assign *value* at a dotted path in *target*, creating intermediate dicts."""
    current = target
    for part in path[:-1]:
        existing = current.get(part)
        if existing is None:
            existing = {}
            current[part] = existing
        if not isinstance(existing, dict):
            raise ValueError(f"cannot assign nested override under non-mapping key {part!r}")
        current = existing
    current[path[-1]] = value


# ---------------------------------------------------------------------------
# Deep merge (adapted from ElysiaRobot native_loader.py)
# ---------------------------------------------------------------------------


def _deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge *overrides* into *base*.

    - Mapping + Mapping → recursive merge
    - list + Mapping → index-based merge (``list[0]`` = ``{"0": ...}``)
    - Otherwise → replace
    """
    merged = dict(base)
    for key, value in overrides.items():
        old_value = merged.get(key)
        if isinstance(old_value, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(old_value, value)
        elif isinstance(old_value, list) and isinstance(value, Mapping):
            merged[key] = _merge_list_by_index(old_value, value)
        else:
            merged[key] = value
    return merged


def _merge_list_by_index(base: list[Any], overrides: Mapping[str, Any]) -> list[Any]:
    """Merge ``phases.0.steps=...``-style overrides into a list."""
    merged = list(base)
    for raw_index, value in overrides.items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            raise ValueError(f"list override index must be an integer, got {raw_index!r}") from None
        if index < 0:
            raise ValueError(f"list override index must be non-negative, got {index}")
        while len(merged) <= index:
            merged.append({})
        old_value = merged[index]
        if isinstance(old_value, Mapping) and isinstance(value, Mapping):
            merged[index] = _deep_merge(old_value, value)
        else:
            merged[index] = value
    return merged


# ---------------------------------------------------------------------------
# Strict key validation with did-you-mean suggestions
# ---------------------------------------------------------------------------


def _validate_config_mapping(mapping: Mapping[str, Any]) -> None:
    """Recursively validate all keys in the final merged config."""
    _validate_node(mapping, ())


def _validate_node(value: Any, path: tuple[str, ...]) -> None:
    if not isinstance(value, Mapping):
        return
    # Open prefixes: skip strict validation (plugin-owned keys)
    if any(path[: len(prefix)] == prefix for prefix in _OPEN_PREFIXES):
        return
    allowed = _CV_FIELD_MAP.get(path)
    if allowed is None and not path:
        allowed = _CV_TOP_LEVEL_SECTIONS - {"_base_"}
    if allowed is None:
        return
    for key, child in value.items():
        key_text = str(key)
        if key_text not in allowed:
            dotted = ".".join((*path, key_text))
            suggestion = _closest_key(key_text, allowed)
            suffix = f"; did you mean {suggestion!r}?" if suggestion else ""
            raise ValueError(f"unknown config key: {dotted!r}{suffix}")
        _validate_node(child, (*path, key_text))


def _closest_key(key: str, candidates: frozenset[str]) -> str | None:
    import difflib
    matches = difflib.get_close_matches(key, sorted(candidates), n=1, cutoff=0.72)
    return matches[0] if matches else None