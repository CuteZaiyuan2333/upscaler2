"""Device helpers for CUDA, Intel XPU, Apple MPS, and CPU."""

from __future__ import annotations

import contextlib
from typing import Iterator

import torch

_IPEX_IMPORTED = False


def _try_import_ipex() -> bool:
    global _IPEX_IMPORTED
    if _IPEX_IMPORTED:
        return True
    try:
        import intel_extension_for_pytorch  # noqa: F401

        _IPEX_IMPORTED = True
        return True
    except ImportError:
        return False


def is_xpu_available() -> bool:
    if not hasattr(torch, "xpu"):
        return False
    try:
        return torch.xpu.is_available()
    except Exception:
        return False


def is_mps_available() -> bool:
    if not hasattr(torch.backends, "mps"):
        return False
    try:
        return torch.backends.mps.is_available()
    except Exception:
        return False


def resolve_device(requested: str = "auto") -> torch.device:
    """
    Pick the best available device.

    Priority: explicit name > cuda > xpu > mps > cpu.
    Use ``--device auto`` (default) for automatic selection.
    """
    requested = requested.lower().strip()

    if requested not in ("auto", "cuda", "xpu", "mps", "cpu"):
        raise ValueError(f"Unsupported device: {requested}")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")

    if requested == "xpu":
        if not is_xpu_available():
            hint = "Install Intel Extension for PyTorch: pip install intel-extension-for-pytorch"
            raise RuntimeError(f"XPU requested but not available. {hint}")
        _try_import_ipex()
        return torch.device("xpu")

    if requested == "mps":
        if not is_mps_available():
            raise RuntimeError("MPS requested but not available on this Mac.")
        return torch.device("mps")

    if requested == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if is_xpu_available():
        _try_import_ipex()
        return torch.device("xpu")
    if is_mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    parts = [f"type={device.type}"]
    if device.type == "cuda" and torch.cuda.is_available():
        parts.append(f"name={torch.cuda.get_device_name(device)}")
    elif device.type == "xpu" and is_xpu_available():
        parts.append(f"count={torch.xpu.device_count()}")
    elif device.type == "mps":
        parts.append("backend=Metal")
    return ", ".join(parts)


def supports_amp(device: torch.device) -> bool:
    return device.type in ("cuda", "xpu", "mps")


@contextlib.contextmanager
def autocast_context(device: torch.device, enabled: bool = True) -> Iterator[None]:
    if not enabled or not supports_amp(device):
        yield
        return

    device_type = device.type
    with torch.autocast(device_type=device_type, dtype=_autocast_dtype(device)):
        yield


def _autocast_dtype(device: torch.device) -> torch.dtype:
    if device.type == "mps":
        return torch.float16
    if device.type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_bf16_supported():
        return torch.bfloat16
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def create_grad_scaler(device: torch.device, enabled: bool = True):
    if not enabled or device.type not in ("cuda", "xpu"):
        return None
    device_type = device.type
    if device.type == "xpu" and not hasattr(torch, "xpu"):
        return None
    try:
        return torch.amp.GradScaler(device_type)
    except Exception:
        return None


def empty_cache(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.type == "xpu" and is_xpu_available():
        torch.xpu.empty_cache()
