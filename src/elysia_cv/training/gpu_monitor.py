"""GPU state sampling for training logs — copied from ElysiaRobot.

Provides per-step GPU memory / utilization sampling and process-level
resource monitoring (RSS, open-fd, cgroup memory).  All sampling is
best-effort: failures are silently swallowed so training is never
interrupted by monitoring failures.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import os
from pathlib import Path
import resource
import subprocess

import torch


BYTES_PER_GB = 1024**3
MIB_PER_GB = 1024


@dataclass
class GpuMetricWindow:
    """Sliding-window aggregator for GPU metrics."""

    max_samples: int = 10
    samples: deque[dict[str, float | int]] = field(init=False)

    def __post_init__(self) -> None:
        self.max_samples = max(int(self.max_samples), 1)
        self.samples = deque(maxlen=self.max_samples)

    def add_sample(self, metrics: dict[str, float | int]) -> dict[str, float | int]:
        """Add a sample and return enriched values with window means."""
        if not metrics:
            return metrics
        self.samples.append(dict(metrics))
        enriched: dict[str, float | int] = dict(metrics)
        for key in (
            "gpu_utilization_pct",
            "gpu_memory_used_gb",
            "gpu_memory_allocated_gb",
            "gpu_memory_reserved_gb",
        ):
            values = [float(sample[key]) for sample in self.samples if key in sample]
            if values:
                enriched[f"{key}_window_mean"] = round(sum(values) / len(values), 3)
        return enriched


def collect_gpu_metrics() -> dict[str, float | int]:
    """Sample current GPU memory and utilization.

    Sampling failures are silently swallowed.
    """
    metrics: dict[str, float | int] = {}
    if not _cuda_available():
        return metrics

    device_index = _current_device_index()
    metrics["gpu_device_index"] = device_index
    allocated = _safe_cuda_memory("memory_allocated", device_index)
    reserved = _safe_cuda_memory("memory_reserved", device_index)
    peak_allocated = _safe_cuda_memory("max_memory_allocated", device_index)
    if allocated is not None:
        metrics["gpu_memory_allocated_gb"] = round(allocated / BYTES_PER_GB, 3)
    if reserved is not None:
        metrics["gpu_memory_reserved_gb"] = round(reserved / BYTES_PER_GB, 3)
    if peak_allocated is not None:
        metrics["gpu_memory_peak_allocated_gb"] = round(peak_allocated / BYTES_PER_GB, 3)

    metrics.update(_nvidia_smi_metrics(device_index))
    return metrics


def collect_process_resource_metrics() -> dict[str, float | int]:
    """Sample light-weight process resources for the current rank.

    Never interrupts training on failure.
    """
    metrics: dict[str, float | int] = {}
    try:
        metrics["process_peak_rss_gb"] = round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3
        )
    except (OSError, ValueError):
        pass
    try:
        metrics["process_open_fds"] = len(tuple(Path("/proc/self/fd").iterdir()))
    except OSError:
        pass
    try:
        soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft_limit > 0 and soft_limit != resource.RLIM_INFINITY:
            metrics["process_open_fd_limit"] = int(soft_limit)
            if "process_open_fds" in metrics:
                metrics["process_open_fd_fraction"] = round(
                    float(metrics["process_open_fds"]) / soft_limit, 6
                )
    except (OSError, ValueError):
        pass
    metrics.update(_process_tree_resource_metrics())
    metrics.update(_cgroup_memory_metrics())
    if _cuda_available():
        device_index = _current_device_index()
        peak_allocated = _safe_cuda_memory("max_memory_allocated", device_index)
        if peak_allocated is not None:
            metrics["gpu_memory_peak_allocated_gb"] = round(peak_allocated / BYTES_PER_GB, 3)
    return metrics


# ------------------------------------------------------------------
# Internal helpers — identical to ElysiaRobot
# ------------------------------------------------------------------


def _process_tree_resource_metrics(
    *,
    pid: int | None = None,
    proc_root: Path = Path("/proc"),
) -> dict[str, float | int]:
    root_identity: str | int = int(pid) if pid is not None else "self"
    pending: list[str | int] = [root_identity]
    seen: set[str | int] = set()
    rss_bytes = 0
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        process_dir = proc_root / str(current)
        try:
            fields = (process_dir / "statm").read_text(encoding="utf-8").split()
            rss_bytes += int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
        except (IndexError, OSError, ValueError):
            pass
        pending.extend(_proc_children(process_dir))
    if not rss_bytes:
        return {}
    return {
        "process_tree_rss_gb": round(rss_bytes / BYTES_PER_GB, 3),
        "process_child_count": max(len(seen) - 1, 0),
    }


def _proc_children(process_dir: Path) -> list[int]:
    children: set[int] = set()
    try:
        task_dirs = tuple((process_dir / "task").iterdir())
    except OSError:
        return []
    for task_dir in task_dirs:
        try:
            children.update(
                int(value)
                for value in (task_dir / "children").read_text(encoding="utf-8").split()
            )
        except (OSError, ValueError):
            continue
    return list(children)


def _cgroup_memory_metrics(*, cgroup_root: Path = Path("/sys/fs/cgroup")) -> dict[str, float]:
    current = _read_memory_bytes(
        cgroup_root / "memory.current",
        cgroup_root / "memory" / "memory.usage_in_bytes",
    )
    limit = _read_memory_bytes(
        cgroup_root / "memory.max",
        cgroup_root / "memory" / "memory.limit_in_bytes",
    )
    metrics: dict[str, float] = {}
    if current is not None:
        metrics["cgroup_memory_current_gb"] = round(current / BYTES_PER_GB, 3)
    if limit is not None and limit < 2**60:
        metrics["cgroup_memory_limit_gb"] = round(limit / BYTES_PER_GB, 3)
        if current is not None and limit > 0:
            metrics["cgroup_memory_fraction"] = round(current / limit, 6)
    return metrics


def _read_memory_bytes(*paths: Path) -> int | None:
    for path in paths:
        try:
            raw = path.read_text(encoding="utf-8").strip()
            if raw != "max":
                return int(raw)
        except (OSError, ValueError):
            continue
    return None


def _cuda_available() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _current_device_index() -> int:
    try:
        return int(torch.cuda.current_device())
    except Exception:
        return 0


def _safe_cuda_memory(method_name: str, device_index: int) -> int | None:
    try:
        method = getattr(torch.cuda, method_name)
        return int(method(device_index))
    except Exception:
        return None


def _nvidia_smi_metrics(device_index: int) -> dict[str, float]:
    query = "utilization.gpu,memory.used,memory.total"
    command = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]
    physical_id = _visible_physical_gpu_id(device_index)
    if physical_id is not None:
        command.insert(1, f"--id={physical_id}")
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=1.0)
    except Exception:
        return {}

    row = _select_nvidia_smi_row(
        completed.stdout, device_index=0 if physical_id is not None else device_index
    )
    if row is None:
        return {}
    parsed = _parse_nvidia_smi_row(row)
    if parsed is None:
        return {}
    utilization_pct, memory_used_mib, memory_total_mib = parsed
    return {
        "gpu_utilization_pct": float(utilization_pct),
        "gpu_memory_used_gb": round(float(memory_used_mib) / MIB_PER_GB, 3),
        "gpu_memory_total_gb": round(float(memory_total_mib) / MIB_PER_GB, 3),
    }


def _visible_physical_gpu_id(device_index: int) -> str | None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return None
    entries = [entry.strip() for entry in visible.split(",") if entry.strip()]
    if device_index < 0 or device_index >= len(entries):
        return None
    return entries[device_index]


def _select_nvidia_smi_row(stdout: str, *, device_index: int) -> str | None:
    rows = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not rows:
        return None
    if 0 <= device_index < len(rows):
        return rows[device_index]
    return rows[0]


def _parse_nvidia_smi_row(row: str) -> tuple[float, float, float] | None:
    parts = [part.strip() for part in row.split(",")]
    if len(parts) < 3:
        return None
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        return None