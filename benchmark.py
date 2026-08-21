"""Benchmark inference/training memory usage and model weight size."""

from __future__ import annotations

import argparse
import gc
import io
import os
import resource
import sys
import time

import torch
import torch.nn.functional as F

from models import SemanticTransferUNet
from utils import autocast_context, describe_device, empty_cache, resolve_device, supports_amp


def _process_rss_bytes() -> int:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss
    except ImportError:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # macOS reports bytes; Linux reports kilobytes
        if sys.platform == "darwin":
            return usage.ru_maxrss
        return usage.ru_maxrss * 1024


def format_bytes(num_bytes: int) -> str:
    if num_bytes >= 1024**3:
        return f"{num_bytes / 1024**3:.2f} GB ({num_bytes:,} B)"
    if num_bytes >= 1024**2:
        return f"{num_bytes / 1024**2:.2f} MB ({num_bytes:,} B)"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.2f} KB ({num_bytes:,} B)"
    return f"{num_bytes} B"


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_weight_bytes(model: torch.nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def checkpoint_bytes(model: torch.nn.Module) -> int:
    buffer = io.BytesIO()
    torch.save({"model_state_dict": model.state_dict()}, buffer)
    return buffer.tell()


class MemoryProbe:
    """Track peak device memory (CUDA/XPU) and process RSS."""

    def __init__(self, device: torch.device):
        self.device = device
        self.rss_start = _process_rss_bytes()
        self.rss_peak = self.rss_start
        self._supports_device_peak = device.type in ("cuda", "xpu")

    def reset(self) -> None:
        gc.collect()
        empty_cache(self.device)
        self.rss_start = _process_rss_bytes()
        self.rss_peak = self.rss_start

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        elif self.device.type == "xpu" and hasattr(torch, "xpu"):
            torch.xpu.reset_peak_memory_stats(self.device)

    def update(self) -> None:
        self.rss_peak = max(self.rss_peak, _process_rss_bytes())

    def device_peak_bytes(self) -> int | None:
        if self.device.type == "cuda":
            self.update()
            return torch.cuda.max_memory_allocated(self.device)
        if self.device.type == "xpu" and hasattr(torch, "xpu"):
            self.update()
            return torch.xpu.max_memory_allocated(self.device)
        if self.device.type == "mps" and hasattr(torch, "mps"):
            self.update()
            if hasattr(torch.mps, "current_allocated_memory"):
                return torch.mps.current_allocated_memory()
        return None

    def process_delta_bytes(self) -> int:
        self.update()
        return self.rss_peak - self.rss_start

    def summary_label(self) -> str:
        if self._supports_device_peak:
            return "device_peak"
        if self.device.type == "mps":
            return "process_rss_delta (MPS 无精确显存峰值 API，使用进程内存增量)"
        return "process_rss_delta"


def make_random_batch(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image = torch.randn(
        batch_size,
        3,
        SemanticTransferUNet.INPUT_SIZE,
        SemanticTransferUNet.INPUT_SIZE,
        device=device,
        dtype=dtype,
    )
    ref = torch.randn(
        batch_size,
        3,
        SemanticTransferUNet.REF_SIZE,
        SemanticTransferUNet.REF_SIZE,
        device=device,
        dtype=dtype,
    )
    target = torch.randn_like(image)
    return image, ref, target


@torch.no_grad()
def run_inference(
    model: SemanticTransferUNet,
    image: torch.Tensor,
    ref: torch.Tensor,
    use_amp: bool,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    with autocast_context(device, enabled=use_amp):
        return model(image, ref)


def run_training_step(
    model: SemanticTransferUNet,
    optimizer: torch.optim.Optimizer,
    image: torch.Tensor,
    ref: torch.Tensor,
    target: torch.Tensor,
    use_amp: bool,
    device: torch.device,
) -> torch.Tensor:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(device, enabled=use_amp):
        pred = model(image, ref)
        loss = F.l1_loss(pred, target) + 0.1 * F.mse_loss(pred, target)
    loss.backward()
    optimizer.step()
    return loss


def benchmark_inference(
    model: SemanticTransferUNet,
    batch_size: int,
    use_amp: bool,
    device: torch.device,
) -> dict[str, float | int | None]:
    probe = MemoryProbe(device)
    probe.reset()

    model.to(device)
    probe.update()

    image, ref, _ = make_random_batch(batch_size, device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize(device)

    t0 = time.perf_counter()
    output = run_inference(model, image, ref, use_amp, device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize(device)

    elapsed = time.perf_counter() - t0

    return {
        "elapsed_sec": elapsed,
        "device_peak_bytes": probe.device_peak_bytes(),
        "process_delta_bytes": probe.process_delta_bytes(),
        "output_shape": tuple(output.shape),
    }


def benchmark_training(
    model: SemanticTransferUNet,
    batch_size: int,
    use_amp: bool,
    device: torch.device,
    warmup_steps: int,
) -> dict[str, float | int | None]:
    probe = MemoryProbe(device)
    probe.reset()

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    image, ref, target = make_random_batch(batch_size, device)

    # First step creates AdamW momentum/variance buffers.
    run_training_step(model, optimizer, image, ref, target, use_amp, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize(device)

    for _ in range(max(0, warmup_steps - 1)):
        run_training_step(model, optimizer, image, ref, target, use_amp, device)

    probe.reset()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize(device)

    t0 = time.perf_counter()
    loss_value = run_training_step(model, optimizer, image, ref, target, use_amp, device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize(device)

    elapsed = time.perf_counter() - t0

    return {
        "elapsed_sec": elapsed,
        "steps": 1 + max(0, warmup_steps - 1) + 1,
        "measured_steps": 1,
        "last_loss": float(loss_value.item()),
        "device_peak_bytes": probe.device_peak_bytes(),
        "process_delta_bytes": probe.process_delta_bytes(),
    }


def print_report(
    device: torch.device,
    model: SemanticTransferUNet,
    infer_stats: dict,
    train_stats: dict,
    use_amp: bool,
    batch_size: int,
) -> None:
    total_params, trainable_params = count_parameters(model)
    weight_bytes = model_weight_bytes(model)
    ckpt_bytes = checkpoint_bytes(model)
    probe = MemoryProbe(device)

    print("=" * 60)
    print("SemanticTransferUNet Benchmark")
    print("=" * 60)
    print(f"Device:         {describe_device(device)}")
    print(f"Batch size:     {batch_size}")
    print(f"Mixed precision:{use_amp}")
    print()
    print("[Model Weights]")
    print(f"  Parameters:        {total_params:,} total / {trainable_params:,} trainable")
    print(f"  Weight memory:     {format_bytes(weight_bytes)}")
    print(f"  Checkpoint (.pt):  {format_bytes(ckpt_bytes)}")
    print()
    print("[Forward Inference]")
    print(f"  Output shape:      {infer_stats['output_shape']}")
    print(f"  Time:              {infer_stats['elapsed_sec']:.3f} s")
    device_peak = infer_stats["device_peak_bytes"]
    if device_peak is not None:
        print(f"  {probe.summary_label()}: {format_bytes(device_peak)}")
    print(f"  Process RSS delta: {format_bytes(infer_stats['process_delta_bytes'])}")
    print()
    print("[Training Step (forward + backward + AdamW)]")
    print(f"  Warmup + measured: {train_stats['steps']} steps ({train_stats['measured_steps']} measured)")
    print(f"  Last loss:         {train_stats['last_loss']:.6f}")
    print(f"  Time:              {train_stats['elapsed_sec']:.3f} s")
    device_peak = train_stats["device_peak_bytes"]
    if device_peak is not None:
        print(f"  {probe.summary_label()}: {format_bytes(device_peak)}")
    print(f"  Process RSS delta: {format_bytes(train_stats['process_delta_bytes'])}")
    print("=" * 60)

    if device.type == "mps":
        print("Note: MPS 上建议使用 Process RSS delta 作为内存参考。")
    if device.type == "cpu":
        print("Note: CPU 模式下显示的是进程内存增量，不是显存。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark memory and weights for SemanticTransferUNet")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "xpu", "mps", "cpu"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--context-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-warmup-steps", type=int, default=2, help="Optimizer states appear after step 1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    use_amp = args.amp and supports_amp(device)

    print(f"Preparing random model and data on {describe_device(device)} ...")

    model = SemanticTransferUNet(
        base_channels=args.base_channels,
        context_dim=args.context_dim,
        num_heads=args.num_heads,
    )

    infer_stats = benchmark_inference(model, args.batch_size, use_amp, device)

    train_model = SemanticTransferUNet(
        base_channels=args.base_channels,
        context_dim=args.context_dim,
        num_heads=args.num_heads,
    )
    train_stats = benchmark_training(
        train_model,
        args.batch_size,
        use_amp,
        device,
        warmup_steps=args.train_warmup_steps,
    )

    print_report(device, model, infer_stats, train_stats, use_amp, args.batch_size)


if __name__ == "__main__":
    main()
