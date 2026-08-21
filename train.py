"""Training script for SemanticTransferUNet."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data import FolderSemanticTransferDataset
from models import SemanticTransferUNet
from utils import (
    autocast_context,
    create_grad_scaler,
    describe_device,
    empty_cache,
    resolve_device,
    supports_amp,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    l1_weight: float,
    mse_weight: float,
) -> torch.Tensor:
    loss = torch.zeros((), device=pred.device, dtype=pred.dtype)
    if l1_weight > 0:
        loss = loss + l1_weight * F.l1_loss(pred, target)
    if mse_weight > 0:
        loss = loss + mse_weight * F.mse_loss(pred, target)
    return loss


@torch.no_grad()
def evaluate(
    model: SemanticTransferUNet,
    loader: DataLoader,
    device: torch.device,
    l1_weight: float,
    mse_weight: float,
    use_amp: bool,
) -> float:
    model.eval()
    total = 0.0

    for batch in loader:
        inp = batch["input"].to(device, non_blocking=True)
        ref = batch["ref"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        with autocast_context(device, enabled=use_amp):
            pred = model(inp, ref)
            loss = compute_loss(pred, target, l1_weight, mse_weight)

        total += loss.item()

    return total / len(loader)


def train_one_epoch(
    model: SemanticTransferUNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    l1_weight: float,
    mse_weight: float,
    use_amp: bool,
    grad_scaler: torch.amp.GradScaler | None,
    grad_clip: float,
    log_interval: int,
) -> float:
    model.train()
    total_loss = 0.0

    progress = tqdm(loader, desc=f"Train epoch {epoch}")
    for step, batch in enumerate(progress, start=1):
        inp = batch["input"].to(device, non_blocking=True)
        ref = batch["ref"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, enabled=use_amp):
            pred = model(inp, ref)
            loss = compute_loss(pred, target, l1_weight, mse_weight)

        if grad_scaler is not None:
            grad_scaler.scale(loss).backward()
            if grad_clip > 0:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        if step % log_interval == 0:
            progress.set_postfix(loss=f"{loss.item():.6f}")

    return total_loss / len(loader)


def save_checkpoint(
    path: Path,
    model: SemanticTransferUNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float | None,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "args": vars(args),
        },
        path,
    )


def build_dataloaders(
    args: argparse.Namespace,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader | None, int]:
    dataset = FolderSemanticTransferDataset(root=args.dataset_dir)
    total_samples = len(dataset)

    if args.val_ratio > 0 and total_samples >= 2:
        val_len = max(1, int(total_samples * args.val_ratio))
        train_len = total_samples - val_len
        generator = torch.Generator().manual_seed(args.seed)
        train_set, val_set = random_split(dataset, [train_len, val_len], generator=generator)
        val_loader: DataLoader | None = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
    else:
        train_set = dataset
        val_loader = None

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, total_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SemanticTransferUNet")
    parser.add_argument("--dataset-dir", type=str, default="dataset")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--l1-weight", type=float, default=1.0)
    parser.add_argument("--mse-weight", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--context-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "xpu", "mps", "cpu"],
        help="Training device. auto = cuda > xpu > mps > cpu",
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--resume", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    use_amp = args.amp and supports_amp(device)
    pin_memory = device.type in ("cuda", "xpu")
    if device.type == "mps":
        pin_memory = False

    print(f"Using device: {describe_device(device)}")
    print(f"Mixed precision: {use_amp}")

    train_loader, val_loader, total_samples = build_dataloaders(args, pin_memory)
    print(f"Dataset: {args.dataset_dir} ({total_samples} samples)")

    model = SemanticTransferUNet(
        base_channels=args.base_channels,
        context_dim=args.context_dim,
        num_heads=args.num_heads,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    grad_scaler = create_grad_scaler(device, enabled=use_amp)

    start_epoch = 1
    best_val = float("inf")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("val_loss", best_val) or best_val
        print(f"Resumed from {args.resume} (epoch {start_epoch - 1})")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            args.l1_weight,
            args.mse_weight,
            use_amp,
            grad_scaler,
            args.grad_clip,
            args.log_interval,
        )

        val_loss: float | None = None
        if val_loader is not None:
            val_loss = evaluate(
                model,
                val_loader,
                device,
                args.l1_weight,
                args.mse_weight,
                use_amp,
            )

        msg = f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.6f}"
        if val_loss is not None:
            msg += f"  val_loss={val_loss:.6f}"
        print(msg)

        if epoch % args.save_every == 0:
            ckpt_path = save_dir / f"epoch_{epoch:04d}.pt"
            save_checkpoint(ckpt_path, model, optimizer, epoch, train_loss, val_loss, args)
            print(f"Saved: {ckpt_path}")

        if val_loss is not None and val_loss < best_val:
            best_val = val_loss
            best_path = save_dir / "best.pt"
            save_checkpoint(best_path, model, optimizer, epoch, train_loss, val_loss, args)
            print(f"New best model: {best_path}")

        empty_cache(device)

    final_path = save_dir / "last.pt"
    save_checkpoint(final_path, model, optimizer, args.epochs, train_loss, val_loss, args)
    print(f"Training finished. Last checkpoint: {final_path}")


if __name__ == "__main__":
    main()
