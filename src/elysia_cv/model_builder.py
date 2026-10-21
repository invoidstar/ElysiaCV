# ═══════════════════════════════════════════════════════════════════
# model_builder.py — SINGLE SOURCE OF TRUTH for all model routing
#
# Every model type is registered ONCE in ``_MODEL_REGISTRY``.
# add a new model → add ONE line here — nothing else changes.
#
# Also handles config-schema normalization (see _cfg_to_dict) so
# that ``_build_video_model`` never needs a manual key whitelist.
# ═══════════════════════════════════════════════════════════════════

from __future__ import annotations

import importlib
from typing import Any

import torch.nn as nn

# ═══════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════

def _cfg_to_dict(section: Any, *, exclude: tuple[str, ...] = ()) -> dict[str, Any]:
    """Convert a config section (SimpleNamespace or similar) to a plain dict.

    Only public, non-callable, non-None attributes are included.
    This eliminates the need for a manual key whitelist in every builder.
    """
    result: dict[str, Any] = {}
    for k, v in section.__dict__.items():
        if k.startswith("_") or callable(v) or v is None or k in exclude:
            continue
        result[k] = v
    return result


# ═══════════════════════════════════════════════════════════════════
# Model registry — one row per model type
#
# Format:  type_name → (module_path, build_func, flags)
#
# flags:
#   "video"         → action recognition (needs _is_video + _forward_train_fn)
#   "yolo"          → YOLO detection (needs _is_yolo + _forward_train_fn)
#   "dict_backbone" → backbone/neck/head must be wrapped in dict()
#   "plugin"        → resolved via ModelRegistry (pre-populated at import)
# ═══════════════════════════════════════════════════════════════════

_MODEL_REGISTRY: dict[str, tuple[str, str, dict[str, Any]]] = {}

def _register(type_name: str, module_path: str, build_func: str = "build_model",
              **flags: Any) -> None:
    """Register a model type."""
    _MODEL_REGISTRY[type_name] = (module_path, build_func, flags)


# -- Video (action recognition) -------------------------------------------------
for _alias, _mod in [
    ("tsn",         "elysia_cv.models.action_recognition.tsn.runtime"),
    ("tsm",         "elysia_cv.models.action_recognition.tsm.runtime"),
    ("mvit",        "elysia_cv.models.action_recognition.mvit.runtime"),
    ("r3d",         "elysia_cv.models.action_recognition.torchvision_video.runtime"),
    ("mc3",         "elysia_cv.models.action_recognition.torchvision_video.runtime"),
    ("r2plus1d",    "elysia_cv.models.action_recognition.torchvision_video.runtime"),
    ("s3d",         "elysia_cv.models.action_recognition.torchvision_video.runtime"),
    ("x3d",         "elysia_cv.models.action_recognition.x3d.runtime"),
    ("slowfast",    "elysia_cv.models.action_recognition.slowfast.runtime"),
    ("xclip",       "elysia_cv.models.action_recognition.xclip.runtime"),
    ("st_adapter",  "elysia_cv.models.action_recognition.st_adapter.runtime"),
    ("stadapter",   "elysia_cv.models.action_recognition.st_adapter.runtime"),
    ("uniformerv2", "elysia_cv.models.action_recognition.uniformerv2.runtime"),
    ("uniformer",   "elysia_cv.models.action_recognition.uniformerv2.runtime"),
    ("timesformer", "elysia_cv.models.action_recognition.timesformer.runtime"),
    ("videomae",    "elysia_cv.models.action_recognition.videomae.runtime"),
]:
    _register(_alias, _mod, family="video")

# -- Detection ------------------------------------------------------------------
_register("fcos", "elysia_cv.models.detection.fcos.runtime",  family="detection")
_register("yolo", "elysia_cv.models.detection.yolo.runtime",  family="yolo")


# ═══════════════════════════════════════════════════════════════════
# family-specific builders
# ═══════════════════════════════════════════════════════════════════

def _build_video_model(model_cfg_section: Any,
                       build_fn,
                       forward_train_fn,
                       ) -> nn.Module:
    """Build any action-recognition model.

    Config params are extracted via :func:`_cfg_to_dict` — no manual
    key whitelist needed.
    """
    model_cfg_dict = _cfg_to_dict(model_cfg_section, exclude=("type",))

    # Normalise backbone: dict → string
    backbone_val = model_cfg_dict.pop("backbone", "")
    if isinstance(backbone_val, dict):
        backbone_val = backbone_val.get("type", "")
    model_cfg_dict["backbone"] = str(backbone_val) if backbone_val else ""

    model = build_fn({"model": model_cfg_dict})
    model._is_video = True
    model._forward_train_fn = forward_train_fn
    return model


def _build_detection_model(model_cfg_section: Any,
                           build_fn,
                           ) -> nn.Module:
    """Build a detection model (FCOS etc)."""
    model_cfg: dict[str, Any] = {
        "type":     model_cfg_section.type,
        "backbone": dict(model_cfg_section.backbone),
    }
    neck = getattr(model_cfg_section, "neck", None)
    if neck is not None:
        model_cfg["neck"] = dict(neck)
    head = getattr(model_cfg_section, "head", None)
    if head is not None:
        model_cfg["head"] = dict(head)
    pretrained_path = getattr(model_cfg_section, "pretrained_path", None)
    if pretrained_path:
        model_cfg["pretrained_path"] = pretrained_path
    return build_fn({"model": model_cfg})


def _build_yolo_model(model_cfg_section: Any,
                      build_fn,
                      forward_train_fn,
                      ) -> nn.Module:
    """Build a YOLO detection model."""
    variant     = str(getattr(model_cfg_section, "variant",     "yolov11n"))
    num_classes = int(getattr(model_cfg_section, "num_classes", 80))
    pretrained  = bool(getattr(model_cfg_section, "pretrained", False))
    model = build_fn({
        "model": {
            "type":        "yolo",
            "variant":     variant,
            "num_classes": num_classes,
            "pretrained":  pretrained,
        }
    })
    model._is_yolo = True
    model._forward_train_fn = forward_train_fn
    return model


# ═══════════════════════════════════════════════════════════════════
# main entry point
# ═══════════════════════════════════════════════════════════════════

def build_model_from_config(model_cfg_section: Any) -> nn.Module:
    """Build a model from the ``cfg.model`` configuration section.

    → Priority: registry table → plugin discovery → cassava legacy → torchvision
    """
    model_type = model_cfg_section.type

    # ---- 1. Registry-based dispatch ----
    entry = _MODEL_REGISTRY.get(model_type)
    if entry is not None:
        module_path, func_name, flags = entry
        mod = importlib.import_module(module_path)
        build_fn = getattr(mod, func_name)
        family = flags.get("family", "")

        if family == "video":
            forward_train_fn = getattr(mod, "forward_train")
            return _build_video_model(model_cfg_section, build_fn, forward_train_fn)

        if family == "yolo":
            forward_train_fn = getattr(mod, "forward_train")
            return _build_yolo_model(model_cfg_section, build_fn, forward_train_fn)

        if family == "detection":
            return _build_detection_model(model_cfg_section, build_fn)

        # unknown family → just call build_fn(raw config)
        return build_fn({"model": _cfg_to_dict(model_cfg_section, exclude=("type",))})

    # ---- 2. Plugin discovery (ModelRegistry) ----
    try:
        from elysia_cv.models.registry import get_model_registry
        registry = get_model_registry()
        plugin_entry = registry.get(model_type)
        plugin = plugin_entry.plugin
        build_target = plugin.get_model_build_target()
        module_path, func_name = build_target.split(":")
        mod = importlib.import_module(module_path)
        build_fn = getattr(mod, func_name)
        return _build_detection_model(model_cfg_section, build_fn)
    except (KeyError, Exception):
        pass

    # ---- 3. Legacy cassava runtime ----
    try:
        from elysia_cv.models.classification.cassava.runtime import build_model
        model_cfg = {
            "model": {
                "type":     model_type,
                "backbone": dict(model_cfg_section.backbone),
                "head":     dict(getattr(model_cfg_section, "head", {})),
            }
        }
        return build_model(model_cfg)
    except Exception:
        pass

    # ---- 4. torchvision fallback ----
    import torchvision
    if hasattr(torchvision.models, model_type):
        return getattr(torchvision.models, model_type)(pretrained=False)

    raise ValueError(
        f"Unknown model type: {model_type!r}. "
        f"Available: {sorted(_MODEL_REGISTRY)}. "
        f"Register via _MODEL_REGISTRY or install a plugin."
    )