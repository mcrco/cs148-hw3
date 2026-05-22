"""§5 — VLM training on CLEVR.

Usage:
    uv run python scripts/train_vlm.py --config configs/vlm_clevr.yaml \\
        --injection all_patches --mask-mode image_bidir \\
        --freeze-config A
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from basics.lora import LoRALinear
from basics.vit import ViT
from vlm.data import build_clevr_loaders
from vlm.eval import batch_clevr_accuracy
from vlm.model import VisionLanguageModel
from vlm.projector import VisionLanguageProjector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--pretrained-vit",
        type=Path,
        required=True,
        help="Path to CLIP-pretrained ViT checkpoint from §3",
    )
    p.add_argument(
        "--injection",
        choices=["cls", "all_patches", "interleaved"],
        default="all_patches",
    )
    p.add_argument(
        "--mask-mode",
        choices=["causal", "image_bidir"],
        default="causal",
    )
    p.add_argument(
        "--freeze-config",
        choices=["A", "B", "C", "D"],
        default="A",
        help="Per writeup §5.6: A=projector only, B=+decoder LoRA, C=+full decoder, D=all three.",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _tokenize_batch(
    batch: dict[str, Any], tokenizer, injection: str, device: torch.device
) -> dict[str, Any]:
    image_prefix = "<image>\n" if injection == "interleaved" else ""
    prompts = [f"{image_prefix}Question: {question}\nAnswer:" for question in batch["question"]]
    texts = [
        f"{prompt} {answer}{tokenizer.eos_token}"
        for prompt, answer in zip(prompts, batch["answer"])
    ]
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=False,
    )
    prompt_ids = tokenizer(prompts, add_special_tokens=False)["input_ids"]

    labels = encoded["input_ids"].clone()
    for i, ids in enumerate(prompt_ids):
        padding = int((encoded["attention_mask"][i] == 0).sum().item())
        labels[i, padding : padding + len(ids)] = -100
    labels[encoded["attention_mask"] == 0] = -100

    return {
        "images": batch["image"].to(device, non_blocking=True),
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
        "labels": labels.to(device),
        "prompts": prompts,
    }


def _set_trainable(module: nn.Module, trainable: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(trainable)


def _apply_lora_to_decoder(module: nn.Module, rank: int, alpha: float) -> None:
    for param in module.parameters():
        param.requires_grad_(False)

    for child_name, child in module.named_children():
        if child_name in {"q_proj", "v_proj"} and isinstance(child, nn.Linear):
            setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha))
        else:
            _apply_lora_to_decoder(child, rank=rank, alpha=alpha)


def _configure_freezing(
    model: VisionLanguageModel,
    freeze_config: str,
    lora_rank: int,
    lora_alpha: float,
) -> None:
    _set_trainable(model.vit, freeze_config == "D")
    _set_trainable(model.projector, True)

    if freeze_config == "A":
        _set_trainable(model.decoder, False)
    elif freeze_config == "B":
        _apply_lora_to_decoder(model.decoder, rank=lora_rank, alpha=lora_alpha)
    elif freeze_config == "C":
        _set_trainable(model.decoder, True)
    elif freeze_config == "D":
        _set_trainable(model.decoder, True)
    else:
        raise ValueError(f"unknown freeze_config: {freeze_config}")


def _build_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, num_steps: int):
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(float(step + 1) / float(warmup_steps), 1e-8)
        progress = (step - warmup_steps) / max(num_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def _evaluate(
    model: VisionLanguageModel,
    val_dl,
    tokenizer,
    injection: str,
    mask_mode: str,
    device: torch.device,
    max_examples: int,
    generation_cfg: dict[str, Any],
) -> dict[str, float]:
    predictions: list[str] = []
    golds: list[str] = []
    q_types: list[str] = []

    for batch in val_dl:
        images = batch["image"].to(device, non_blocking=True)
        image_prefix = "<image>\n" if injection == "interleaved" else ""
        prompts = [f"{image_prefix}Question: {question}\nAnswer:" for question in batch["question"]]
        preds = model.generate(
            images,
            prompts,
            injection=injection,
            mask_mode=mask_mode,
            max_new_tokens=generation_cfg["max_new_tokens"],
            do_sample=generation_cfg["do_sample"],
            temperature=generation_cfg["temperature"],
            top_p=generation_cfg["top_p"],
        )
        predictions.extend(preds)
        golds.extend(batch["answer"])
        q_types.extend(batch["q_type"])
        if len(golds) >= max_examples:
            break

    predictions = predictions[:max_examples]
    golds = golds[:max_examples]
    q_types = q_types[:max_examples]
    return batch_clevr_accuracy(predictions, golds, q_types=q_types)


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device(args.device)

    if args.output_dir is None:
        args.output_dir = (
            Path("runs") / f"vlm_{args.injection}_{args.mask_mode}_{args.freeze_config}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = cfg["train"]
    optim_cfg = cfg["optim"]
    decoder_cfg = cfg["decoder"]
    generation_cfg = cfg["generation"]
    vit_cfg = cfg["vit"]

    train_dl, val_dl = build_clevr_loaders(
        img_size=vit_cfg["img_size"],
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
    )

    vit_checkpoint = torch.load(args.pretrained_vit, map_location=device, weights_only=True)
    vit = ViT(
        img_size=vit_cfg["img_size"],
        patch_size=vit_cfg["patch_size"],
        d_model=vit_cfg["d_model"],
        num_heads=vit_cfg["num_heads"],
        num_blocks=vit_cfg["num_blocks"],
        dropout=vit_cfg["dropout"],
    ).to(device)
    vit.load_state_dict(vit_checkpoint["vit"])

    tokenizer = AutoTokenizer.from_pretrained(decoder_cfg["model_name"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_name = decoder_cfg["torch_dtype"]
    torch_dtype = getattr(torch, dtype_name) if device.type == "cuda" else torch.float32
    model_kwargs = {"dtype": torch_dtype}
    if device.type == "cuda":
        attn_implementation = decoder_cfg["attn_implementation"]
        if args.mask_mode == "image_bidir" and attn_implementation == "flash_attention_2":
            # FlashAttention-2 cannot consume the custom 4D additive mask used for
            # bidirectional image attention.
            attn_implementation = "eager"
            print("Using eager decoder attention because image_bidir needs a custom 4D mask.")
        model_kwargs["attn_implementation"] = attn_implementation

    decoder = AutoModelForCausalLM.from_pretrained(decoder_cfg["model_name"], **model_kwargs)
    decoder.to(device)

    image_token_id = None
    if args.injection == "interleaved":
        tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
        image_token_id = tokenizer.convert_tokens_to_ids("<image>")
        decoder.resize_token_embeddings(len(tokenizer))

    decoder_dim = decoder.get_input_embeddings().embedding_dim
    projector = VisionLanguageProjector(
        d_image=vit.d_model,
        d_decoder=decoder_dim,
        expansion=cfg["projector"]["expansion"],
    ).to(device=device, dtype=torch_dtype)

    model = VisionLanguageModel(
        vit=vit,
        projector=projector,
        decoder=decoder,
        tokenizer=tokenizer,
        image_token_id=image_token_id,
    ).to(device)
    _configure_freezing(
        model,
        freeze_config=args.freeze_config,
        lora_rank=decoder_cfg["lora"]["rank"],
        lora_alpha=decoder_cfg["lora"]["alpha"],
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=optim_cfg["lr"],
        weight_decay=optim_cfg["weight_decay"],
        betas=tuple(optim_cfg["betas"]),
    )
    scheduler = _build_scheduler(
        optimizer,
        warmup_steps=optim_cfg["warmup_steps"],
        num_steps=train_cfg["num_steps"],
    )

    scaler_enabled = device.type == "cuda" and torch_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    autocast_dtype = torch_dtype if device.type == "cuda" else torch.bfloat16
    grad_accum = train_cfg["gradient_accumulation_steps"]
    best_val_acc = -1.0
    running_loss = 0.0
    last_grad_norm = 0.0

    model.train()
    optimizer.zero_grad(set_to_none=True)
    train_iter = itertools.cycle(train_dl)
    progress = tqdm(range(1, train_cfg["num_steps"] + 1), desc="train_vlm")
    for step in progress:
        batch = next(train_iter)
        inputs = _tokenize_batch(batch, tokenizer, args.injection, device)
        with torch.autocast(
            device_type=device.type, dtype=autocast_dtype, enabled=device.type == "cuda"
        ):
            out = model(
                images=inputs["images"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
                injection=args.injection,
                mask_mode=args.mask_mode,
            )
            loss = out["loss"] / grad_accum

        scaler.scale(loss).backward()
        running_loss += float(loss.detach().cpu()) * grad_accum

        if step % grad_accum == 0:
            scaler.unscale_(optimizer)
            last_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float("inf"))
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if step % train_cfg["log_every"] == 0:
            peak_mem = (
                torch.cuda.max_memory_allocated(device) / (1024**2)
                if device.type == "cuda"
                else 0.0
            )
            avg_loss = running_loss / train_cfg["log_every"]
            running_loss = 0.0
            progress.set_postfix(
                loss=f"{avg_loss:.4f}",
                grad=f"{last_grad_norm:.2f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
                mem_mb=f"{peak_mem:.0f}",
            )
            print(
                f"step {step:05d} train_loss={avg_loss:.4f} "
                f"grad_norm={last_grad_norm:.4f} lr={scheduler.get_last_lr()[0]:.2e} "
                f"peak_mem_mb={peak_mem:.0f}"
            )

        if step % train_cfg["eval_every_steps"] == 0 or step == train_cfg["num_steps"]:
            metrics = _evaluate(
                model,
                val_dl,
                tokenizer,
                args.injection,
                args.mask_mode,
                device,
                max_examples=train_cfg["eval_max_examples"],
                generation_cfg=generation_cfg,
            )
            val_acc = metrics["overall"]
            print(
                f"step {step:05d} val_acc={val_acc:.4f} "
                + " ".join(f"val_{k}={v:.4f}" for k, v in sorted(metrics.items()) if k != "overall")
            )
            model.train()

            checkpoint = {
                "model": model.state_dict(),
                "vit_config": vit_cfg,
                "decoder_model_name": decoder_cfg["model_name"],
                "injection": args.injection,
                "mask_mode": args.mask_mode,
                "freeze_config": args.freeze_config,
                "image_token_id": image_token_id,
                "tokenizer_len": len(tokenizer),
                "step": step,
                "val_metrics": metrics,
                "config": cfg,
            }
            torch.save(checkpoint, args.output_dir / "last.pt")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(checkpoint, args.output_dir / "best.pt")


if __name__ == "__main__":
    main()
