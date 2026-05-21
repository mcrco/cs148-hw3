"""§3 — CLIP-style pretraining on EuroSAT.

You implement the training loop. This script provides the CLI scaffolding,
config loading, and logging hooks.

Usage:
    uv run python scripts/pretrain_clip.py --config configs/clip_eurosat.yaml
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import wandb
import yaml

from basics.text_encoder import FrozenTextEncoder
from basics.vit import ViT
from vlm.clip import ProjectionHeads, clip_loss, init_logit_scale
from vlm.data import EUROSAT_CLASSES, build_eurosat_loaders
from vlm.eval import zeroshot_classification_accuracy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("runs/clip_eurosat"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--pos-enc",
        choices=["learned", "rope_1d", "rope_2d"],
        default=None,
        help="Positional encoding for ViT (default: vit.pos_encoding in config, else learned)",
    )
    p.add_argument(
        "--max-img-size",
        type=int,
        default=None,
        help="Max image side for RoPE caches / learned-PE extrapolation (default: max(train, eval) sizes)",
    )
    p.add_argument(
        "--eval-img-size",
        type=int,
        default=96,
        help="Image size for length-extrapolation zero-shot eval",
    )
    p.add_argument("--wandb", action="store_true", help="Log to W&B")
    return p.parse_args()


def build_vit(vit_cfg: dict, pos_encoding: str, max_img_size: int) -> ViT:
    return ViT(
        vit_cfg["img_size"],
        vit_cfg["patch_size"],
        vit_cfg["d_model"],
        vit_cfg["num_heads"],
        vit_cfg["num_blocks"],
        vit_cfg["dropout"],
        pos_encoding=pos_encoding,
        max_img_size=max_img_size,
    )


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device(args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.wandb:
        wandb.init(project="cs148b-hw3-clip")

    # TODO: students fill in the training loop.
    # Sketch:
    #   1. Build train/val/test loaders via vlm.data.build_eurosat_loaders.
    #   2. Build the ViT (basics.vit.ViT) and FrozenTextEncoder.
    #   3. Build ProjectionHeads + logit_scale.
    #   4. AdamW optimizer, cosine LR schedule.
    #   5. For each epoch:
    #         - Train one epoch with vlm.clip.clip_loss.
    #         - Clamp logit_scale.data to <= ln(100).
    #         - Compute zero-shot val accuracy via vlm.eval.zeroshot_classification_accuracy.
    #         - Log to stdout (and W&B if args.wandb).
    #   6. Save the best checkpoint to args.output_dir / "best.pt".
    vit_cfg = cfg["vit"]
    text_cfg = cfg["text_encoder"]
    proj_cfg = cfg["projection"]
    optim_cfg = cfg["optim"]
    train_cfg = cfg["train"]

    pos_encoding = args.pos_enc or vit_cfg.get("pos_encoding", "learned")
    train_img_size = vit_cfg["img_size"]
    max_img_size = args.max_img_size or max(train_img_size, args.eval_img_size)

    train_dl, val_dl, test_dl = build_eurosat_loaders(
        train_img_size, train_cfg["batch_size"], train_cfg["num_workers"]
    )
    extrap_val_dl, _, _ = build_eurosat_loaders(
        args.eval_img_size, train_cfg["batch_size"], train_cfg["num_workers"]
    )
    model = build_vit(vit_cfg, pos_encoding, max_img_size).to(device)
    text_encoder = FrozenTextEncoder(text_cfg["model_name"]).to(device)
    text_encoder.eval()  # never in train mode
    heads = ProjectionHeads(
        vit_cfg["d_model"], text_encoder.embedding_dim, proj_cfg["d_proj"]
    ).to(device)
    logit_scale = torch.nn.Parameter(init_logit_scale().detach().to(device))
    params = list(heads.parameters()) + list(model.parameters()) + [logit_scale]
    optimizer = torch.optim.AdamW(
        params=params,
        lr=optim_cfg["lr"],
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
        T_max=max(train_cfg["num_epochs"] * len(train_dl) - optim_cfg["warmup_steps"], 1),
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[optim_cfg["warmup_steps"]],
    )

    class_prompts = [f"a satellite image of {name}" for name in EUROSAT_CLASSES]
    class_indices = list(range(len(class_prompts)))
    best_val_acc = -1.0

    for epoch in range(train_cfg["num_epochs"]):
        model.train()
        heads.train()

        train_loss = 0.0
        train_examples = 0
        for i, (images, captions) in enumerate(train_dl):
            images = images.to(device)
            image_embeds = model(images)
            with torch.no_grad():
                text_embeds = text_encoder(captions)  # (B, d_text)
            image_embeds, text_embeds = heads(image_embeds, text_embeds)
            loss = clip_loss(image_embeds, text_embeds, logit_scale)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logit_scale.data.clamp_(max=math.log(100.0))
            scheduler.step()

            batch_size = images.size(0)
            train_loss += loss.item() * batch_size
            train_examples += batch_size

            if args.wandb and i % train_cfg["log_every"] == 0:
                wandb.log({"train/step_loss": loss.item()})

        train_epoch_loss = train_loss / max(train_examples, 1)
        metrics = {
            "epoch": epoch + 1,
            "train/loss": train_epoch_loss,
            "lr": scheduler.get_last_lr()[0],
            "logit_scale": logit_scale.item(),
        }

        if args.wandb:
            wandb.log(metrics)

        print(
            f"epoch {epoch + 1:03d}/{train_cfg['num_epochs']} "
            f"train_loss={train_epoch_loss:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if (epoch + 1) % train_cfg["eval_every_epoch"] == 0:
            val_acc = zeroshot_classification_accuracy(
                model,
                heads,
                text_encoder,
                val_dl,
                class_prompts,
                class_indices,
                device,
            )
            print(f"epoch {epoch + 1:03d} val_zeroshot_acc={val_acc:.4f}")
            if args.wandb:
                wandb.log({"epoch": epoch + 1, "val/zeroshot_acc": val_acc})

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "vit": model.state_dict(),
                        "projection_heads": heads.state_dict(),
                        "logit_scale": logit_scale.detach().cpu(),
                        "epoch": epoch + 1,
                        "val_zeroshot_acc": val_acc,
                        "pos_encoding": pos_encoding,
                        "vit_config": {**vit_cfg, "pos_encoding": pos_encoding, "max_img_size": max_img_size},
                    },
                    args.output_dir / "best.pt",
                )

    checkpoint = torch.load(
        args.output_dir / "best.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["vit"])
    heads.load_state_dict(checkpoint["projection_heads"])
    logit_scale.data.copy_(checkpoint["logit_scale"].to(device))
    test_acc = zeroshot_classification_accuracy(
        model,
        heads,
        text_encoder,
        test_dl,
        class_prompts,
        class_indices,
        device,
    )
    extrap_val_acc = zeroshot_classification_accuracy(
        model,
        heads,
        text_encoder,
        extrap_val_dl,
        class_prompts,
        class_indices,
        device,
    )
    print(
        f"pos_encoding={pos_encoding} "
        f"best_val_zeroshot_acc={best_val_acc:.4f} "
        f"test_zeroshot_acc={test_acc:.4f} "
        f"extrap_val_zeroshot_acc={extrap_val_acc:.4f} "
        f"(eval_img_size={args.eval_img_size})"
    )
    if args.wandb:
        wandb.log(
            {
                "test/zeroshot_acc": test_acc,
                "best_val/zeroshot_acc": best_val_acc,
                "extrap_val/zeroshot_acc": extrap_val_acc,
                "eval_img_size": args.eval_img_size,
            }
        )


if __name__ == "__main__":
    main()
