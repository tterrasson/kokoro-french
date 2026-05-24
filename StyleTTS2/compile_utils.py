"""Utilities for conservative optional torch.compile support."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch


SAFE_MODULES = ("bert_encoder",)

UNSAFE_MODULE_REASONS = {
    "decoder": "training-time random branches and variable audio lengths",
    "text_encoder": "pack_padded_sequence and dynamic CPU lengths",
    "predictor": "pack_padded_sequence and dynamic alignment shapes",
    "bert": "large external model with high trace overhead",
    "diffusion": "dynamic diffusion path; often disabled for Kokoro fine-tuning",
    "mpd": "discriminator path is shape-sensitive and expensive to compile",
    "msd": "STFT discriminator path is shape-sensitive and expensive to compile",
    "msstft": "multi-scale STFT discriminator is shape-sensitive",
    "subband": "subband discriminator is shape-sensitive",
    "wd": "WavLM discriminator is large and expensive to trace",
    "style_encoder": "spectral-norm encoder kept eager by default",
    "predictor_encoder": "spectral-norm encoder kept eager by default",
}


def compile_model_for_training(model, config: Mapping | None, logger=None):
    """Compile allowlisted model forwards in-place without changing checkpoint keys.

    Replacing a module with torch.compile(module) wraps it in an OptimizedModule,
    which changes state_dict prefixes. Compiling only forward keeps the original
    nn.Module object in the model container, so load/export formats stay stable.
    """
    compile_config = (config or {}).get("torch_compile", {})
    if not compile_config or not compile_config.get("enabled", False):
        _log(logger, "info", "torch.compile disabled")
        return model

    if not hasattr(torch, "compile"):
        _log(
            logger,
            "warning",
            "torch.compile requested but unavailable in this PyTorch build",
        )
        return model

    requested_modules = compile_config.get("modules", [])
    modules_to_compile, skipped = _resolve_compile_modules(requested_modules, model)
    _log_requested_modules(logger, requested_modules, modules_to_compile)
    _log_skipped_modules(logger, skipped)

    if not modules_to_compile:
        _log(logger, "info", "torch.compile enabled but no modules selected")
        return model

    backend = compile_config.get("backend", "inductor")
    mode = compile_config.get("mode", "reduce-overhead")
    fullgraph = bool(compile_config.get("fullgraph", False))
    dynamic = compile_config.get("dynamic", True)
    recompile_limit = compile_config.get("recompile_limit", 8)

    if hasattr(torch, "_dynamo"):
        torch._dynamo.config.recompile_limit = recompile_limit

    compiled = []
    failed = []
    for key in modules_to_compile:
        module = model[key]
        try:
            module.forward = torch.compile(
                module.forward,
                backend=backend,
                mode=mode,
                fullgraph=fullgraph,
                dynamic=dynamic,
            )
            compiled.append(key)
        except Exception as exc:
            failed.append(key)
            _log(logger, "warning", f"torch.compile skipped for {key}: {exc}")

    if compiled:
        _log(
            logger,
            "info",
            "torch.compile enabled for modules: " + ", ".join(compiled),
        )
    if failed:
        _log(
            logger,
            "warning",
            "torch.compile failed for modules: " + ", ".join(failed),
        )

    return model


def _resolve_compile_modules(requested_modules, model):
    model_keys = set(model.keys() if hasattr(model, "keys") else model)
    skipped = []

    if requested_modules == "safe":
        requested = list(SAFE_MODULES)
        for key, reason in UNSAFE_MODULE_REASONS.items():
            if key in model_keys:
                skipped.append((key, reason))
    elif requested_modules is None:
        requested = []
    elif isinstance(requested_modules, str):
        requested = [requested_modules]
    elif isinstance(requested_modules, Iterable):
        requested = list(requested_modules)
    else:
        skipped.append(("torch_compile.modules", "expected a list or 'safe' preset"))
        requested = []

    modules_to_compile = []
    seen = set()
    for key in requested:
        if key in seen:
            continue
        seen.add(key)
        if key not in model_keys:
            skipped.append((str(key), "unknown module"))
            continue
        if requested_modules == "safe" and key in UNSAFE_MODULE_REASONS:
            skipped.append((key, UNSAFE_MODULE_REASONS[key]))
            continue
        modules_to_compile.append(key)

    return modules_to_compile, skipped


def _log_requested_modules(logger, requested_modules, modules_to_compile):
    if requested_modules == "safe":
        requested = "safe preset"
    elif isinstance(requested_modules, str):
        requested = requested_modules
    elif requested_modules:
        requested = ", ".join(str(key) for key in requested_modules)
    else:
        requested = "none"

    _log(logger, "info", f"torch.compile requested modules: {requested}")
    if modules_to_compile:
        _log(
            logger,
            "info",
            "torch.compile selected modules: " + ", ".join(modules_to_compile),
        )


def _log_skipped_modules(logger, skipped):
    if not skipped:
        return

    expected_skips = []
    warning_skips = []
    for key, reason in skipped:
        if reason in {"unknown module", "expected a list or 'safe' preset"}:
            warning_skips.append((key, reason))
        else:
            expected_skips.append((key, reason))

    if expected_skips:
        skipped_text = "; ".join(f"{key} ({reason})" for key, reason in expected_skips)
        _log(logger, "info", "torch.compile skipped modules: " + skipped_text)

    if warning_skips:
        skipped_text = "; ".join(f"{key} ({reason})" for key, reason in warning_skips)
        _log(logger, "warning", "torch.compile skipped modules: " + skipped_text)


def _log(logger, level, message):
    if logger is None:
        print(message)
        return

    log_fn = getattr(logger, level, None)
    if log_fn is not None:
        log_fn(message)
    else:
        print(message)
