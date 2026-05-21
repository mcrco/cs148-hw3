"""§4 — Compare full FT, LoRA, and linear probe on RESISC45.

Usage:
    uv run python scripts/finetune_resisc.py --config configs/lora_resisc.yaml \\
        --method lora --rank 8 --pretrained runs/clip_eurosat/best.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from basics.lora import apply_lora_to_attention
from basics.vit import ViT
from vlm.data import build_resisc45_loaders


class ViTClassifier(nn.Module):
    def __init__(self, vit: ViT, num_classes: int) -> None:
        super().__init__()
        self.vit = vit
        self.head = nn.Linear(vit.d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.vit(x))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--clip-config", type=Path, default=Path("configs/clip_eurosat.yaml"))
    p.add_argument("--method", choices=["linear_probe", "lora", "full_ft"], required=True)
    p.add_argument("--rank", type=int, default=8, help="LoRA rank (only for --method lora)")
    p.add_argument("--alpha", type=float, default=16.0, help="LoRA alpha (only for --method lora)")
    p.add_argument(
        "--pretrained",
        type=Path,
        required=True,
        help="Path to CLIP-pretrained ViT checkpoint from §3",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def configure_model(
    model: ViTClassifier,
    method: str,
    rank: int,
    alpha: float,
) -> None:
    if method == "linear_probe":
        for param in model.vit.parameters():
            param.requires_grad_(False)
    elif method == "lora":
        apply_lora_to_attention(model.vit, rank=rank, alpha=alpha)
    elif method == "full_ft":
        for param in model.parameters():
            param.requires_grad_(True)


@torch.no_grad()
def evaluate(model: ViTClassifier, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


def train_one_epoch(
    model: ViTClassifier,
    loader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    criterion = nn.CrossEntropyLoss()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open(args.clip_config) as f:
        clip_cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    vit_cfg = clip_cfg["vit"]
    optim_cfg = cfg["optim"]
    train_cfg = cfg["train"]
    method_cfg = cfg["methods"][args.method]
    lr = method_cfg.get("lr", optim_cfg["lr"])

    if args.output_dir is None:
        if args.method == "lora":
            args.output_dir = Path("runs") / f"resisc_{args.method}_rank{args.rank}"
        else:
            args.output_dir = Path("runs") / f"resisc_{args.method}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_dl, test_dl = build_resisc45_loaders(
        img_size=vit_cfg["img_size"],
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
    )

    checkpoint = torch.load(args.pretrained, map_location=device, weights_only=True)
    vit = ViT(
        img_size=vit_cfg["img_size"],
        patch_size=vit_cfg["patch_size"],
        d_model=vit_cfg["d_model"],
        num_heads=vit_cfg["num_heads"],
        num_blocks=vit_cfg["num_blocks"],
        dropout=vit_cfg["dropout"],
    )
    vit.load_state_dict(checkpoint["vit"])
    model = ViTClassifier(vit, num_classes=cfg["num_classes"])
    configure_model(model, args.method, args.rank, args.alpha)
    model = model.to(device)

    total_params, trainable_params = count_params(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=lr,
        weight_decay=optim_cfg["weight_decay"],
        betas=tuple(optim_cfg["betas"]),
    )
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        total_iters=optim_cfg["warmup_steps"],
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(
            train_cfg["num_epochs"] * len(train_dl) - optim_cfg["warmup_steps"],
            1,
        ),
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[optim_cfg["warmup_steps"]],
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start_time = time.perf_counter()
    best_test_acc = -1.0
    final_test_acc = 0.0

    for epoch in range(train_cfg["num_epochs"]):
        train_loss = train_one_epoch(model, train_dl, optimizer, scheduler, device)

        if (epoch + 1) % train_cfg["eval_every_epoch"] == 0:
            test_acc = evaluate(model, test_dl, device)
            final_test_acc = test_acc
            best_test_acc = max(best_test_acc, test_acc)
            print(
                f"epoch {epoch + 1:03d}/{train_cfg['num_epochs']} "
                f"train_loss={train_loss:.4f} test_acc={test_acc:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

    wall_clock_seconds = time.perf_counter() - start_time
    peak_memory_bytes = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )

    metrics = {
        "method": args.method,
        "rank": args.rank if args.method == "lora" else None,
        "alpha": args.alpha if args.method == "lora" else None,
        "lr": lr,
        "num_epochs": train_cfg["num_epochs"],
        "final_test_acc": final_test_acc,
        "best_test_acc": best_test_acc,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "peak_memory_bytes": peak_memory_bytes,
        "wall_clock_seconds": wall_clock_seconds,
    }

    metrics_path = args.output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(
        f"method={args.method} final_test_acc={final_test_acc:.4f} "
        f"trainable_params={trainable_params:,} "
        f"peak_memory_bytes={peak_memory_bytes:,} "
        f"wall_clock_seconds={wall_clock_seconds:.1f}"
    )
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
