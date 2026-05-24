"""Utilities for optional torch.compile support in training scripts."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def compile_model_for_training(model, config: Mapping | None, logger=None):
    """Compile each model module in-place without changing checkpoint keys.

    Replacing a module with torch.compile(module) wraps it in an OptimizedModule,
    which changes state_dict prefixes. Compiling only forward keeps the original
    nn.Module object in the model container, so load/export formats stay stable.
    """
    compile_config = (config or {}).get("torch_compile", {})
    if not compile_config or not compile_config.get("enabled", False):
        _log(logger, "info", "torch.compile disabled")
        return model

    if not hasattr(torch, "compile"):
        _log(logger, "warning", "torch.compile requested but unavailable in this PyTorch build")
        return model

    backend = compile_config.get("backend", "inductor")
    mode = compile_config.get("mode", "default")
    fullgraph = bool(compile_config.get("fullgraph", False))
    dynamic = compile_config.get("dynamic", True)
    recompile_limit = compile_config.get("recompile_limit", 32)

    if hasattr(torch, "_dynamo"):
        torch._dynamo.config.recompile_limit = recompile_limit

    compiled = []
    failed = []
    for key in model:
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


def _log(logger, level, message):
    if logger is None:
        print(message)
        return

    log_fn = getattr(logger, level, None)
    if log_fn is not None:
        log_fn(message)
    else:
        print(message)
