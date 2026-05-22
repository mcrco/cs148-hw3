"""§5 — Qualitative evaluation of a trained VLM.

Generates predictions on a held-out CLEVR sample and reports per-q_type
accuracy. Useful for both Problem (vlm_qualitative) and Problem (mrope_impl).

Usage:
    uv run python scripts/eval_vlm.py \\
        --checkpoint runs/vlm_all_patches_image_bidir_A/best.pt \\
        --num-examples 10 --save-images
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from basics.lora import LoRALinear
from basics.vit import ViT
from vlm.data import CLEVRMiniDataset
from vlm.eval import batch_clevr_accuracy, clevr_exact_match
from vlm.model import VisionLanguageModel
from vlm.projector import VisionLanguageProjector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--num-examples", type=int, default=10,
                   help="Number of examples to dump for qualitative inspection")
    p.add_argument("--max-eval", type=int, default=500,
                   help="Number of examples to use for accuracy computation")
    p.add_argument("--save-images", action="store_true",
                   help="Save the example images alongside the JSON output")
    p.add_argument("--output-dir", type=Path, default=Path("runs/vlm_qualitative"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for qualitative example sampling")
    return p.parse_args()


class _CLEVRWithImageFile(CLEVRMiniDataset):
    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        item["image_file"] = self.examples[idx]["image_file"]
        return item


def _apply_lora_to_decoder(module: nn.Module, rank: int, alpha: float) -> None:
    for param in module.parameters():
        param.requires_grad_(False)

    for child_name, child in module.named_children():
        if child_name in {"q_proj", "v_proj"} and isinstance(child, nn.Linear):
            setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha))
        else:
            _apply_lora_to_decoder(child, rank=rank, alpha=alpha)


def _load_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[VisionLanguageModel, Any, str, str, dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg = ckpt["config"]
    vit_cfg = ckpt["vit_config"]
    decoder_cfg = cfg["decoder"]
    injection = ckpt["injection"]
    mask_mode = ckpt["mask_mode"]
    freeze_config = ckpt["freeze_config"]
    image_token_id = ckpt.get("image_token_id")

    vit = ViT(
        img_size=vit_cfg["img_size"],
        patch_size=vit_cfg["patch_size"],
        d_model=vit_cfg["d_model"],
        num_heads=vit_cfg["num_heads"],
        num_blocks=vit_cfg["num_blocks"],
        dropout=vit_cfg["dropout"],
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(ckpt["decoder_model_name"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_name = decoder_cfg["torch_dtype"]
    torch_dtype = getattr(torch, dtype_name) if device.type == "cuda" else torch.float32
    model_kwargs: dict[str, Any] = {"dtype": torch_dtype}
    if device.type == "cuda":
        attn_implementation = decoder_cfg["attn_implementation"]
        if mask_mode == "image_bidir" and attn_implementation == "flash_attention_2":
            attn_implementation = "eager"
        model_kwargs["attn_implementation"] = attn_implementation

    decoder = AutoModelForCausalLM.from_pretrained(
        ckpt["decoder_model_name"], **model_kwargs
    )
    decoder.to(device)

    if injection == "interleaved":
        tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
        image_token_id = tokenizer.convert_tokens_to_ids("<image>")
        decoder.resize_token_embeddings(len(tokenizer))

    if freeze_config == "B":
        _apply_lora_to_decoder(
            decoder,
            rank=decoder_cfg["lora"]["rank"],
            alpha=decoder_cfg["lora"]["alpha"],
        )

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
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tokenizer, injection, mask_mode, cfg


def _build_eval_loader(split: str, img_size: int, batch_size: int) -> DataLoader:
    dataset = _CLEVRWithImageFile(split=split, img_size=img_size)

    def _collate(batch: list[dict]) -> dict:
        return {
            "image": torch.stack([b["image"] for b in batch]),
            "question": [b["question"] for b in batch],
            "answer": [b["answer"] for b in batch],
            "q_type": [b["q_type"] for b in batch],
            "image_file": [b["image_file"] for b in batch],
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def _run_eval(
    model: VisionLanguageModel,
    eval_dl: DataLoader,
    injection: str,
    mask_mode: str,
    device: torch.device,
    max_examples: int,
    generation_cfg: dict[str, Any],
) -> list[dict]:
    records: list[dict] = []

    for batch in eval_dl:
        images = batch["image"].to(device, non_blocking=True)
        image_prefix = "<image>\n" if injection == "interleaved" else ""
        prompts = [
            f"{image_prefix}Question: {question}\nAnswer:"
            for question in batch["question"]
        ]
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

        for image_file, question, gold, prediction, q_type in zip(
            batch["image_file"],
            batch["question"],
            batch["answer"],
            preds,
            batch["q_type"],
        ):
            records.append(
                {
                    "image_file": image_file,
                    "question": question,
                    "gold": gold,
                    "prediction": prediction,
                    "correct": clevr_exact_match(prediction, gold),
                    "q_type": q_type,
                }
            )
            if len(records) >= max_examples:
                return records

    return records


def _sample_qualitative_examples(
    records: list[dict], num_examples: int, seed: int
) -> list[dict]:
    if len(records) <= num_examples:
        return list(records)

    rng = random.Random(seed)
    correct = [r for r in records if r["correct"]]
    incorrect = [r for r in records if not r["correct"]]

    n_incorrect = min(len(incorrect), num_examples // 2)
    n_correct = min(len(correct), num_examples - n_incorrect)
    if n_correct + n_incorrect < num_examples:
        remaining = num_examples - n_correct - n_incorrect
        if len(correct) > n_correct:
            n_correct = min(len(correct), n_correct + remaining)
        elif len(incorrect) > n_incorrect:
            n_incorrect = min(len(incorrect), n_incorrect + remaining)

    sample = rng.sample(incorrect, n_incorrect) + rng.sample(correct, n_correct)
    rng.shuffle(sample)
    return sample


def _print_summary(metrics: dict[str, float]) -> None:
    print(f"\n{'metric':<20} {'accuracy':>10}")
    print("-" * 32)
    for key in sorted(metrics, key=lambda k: (k != "overall", k)):
        print(f"{key:<20} {metrics[key]:10.4f}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    model, _tokenizer, injection, mask_mode, cfg = _load_model(args.checkpoint, device)
    generation_cfg = cfg["generation"]
    vit_cfg = cfg["vit"]
    train_cfg = cfg["train"]

    eval_dl = _build_eval_loader(
        split=args.split,
        img_size=vit_cfg["img_size"],
        batch_size=train_cfg["batch_size"],
    )

    records = _run_eval(
        model,
        eval_dl,
        injection,
        mask_mode,
        device,
        max_examples=args.max_eval,
        generation_cfg=generation_cfg,
    )

    metrics = batch_clevr_accuracy(
        [r["prediction"] for r in records],
        [r["gold"] for r in records],
        q_types=[r["q_type"] for r in records],
    )
    _print_summary(metrics)

    qualitative = _sample_qualitative_examples(records, args.num_examples, args.seed)
    clevr_root = Path("data/clevr_mini")
    images_dir = args.output_dir / "images"
    if args.save_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    examples_path = args.output_dir / "examples.jsonl"
    with open(examples_path, "w") as f:
        for row in qualitative:
            src = clevr_root / "images" / row["image_file"]
            if args.save_images:
                dst = images_dir / row["image_file"]
                shutil.copy2(src, dst)
                row = {**row, "saved_image": str(dst)}
            else:
                row = {**row, "saved_image": str(src)}
            f.write(json.dumps(row) + "\n")

    print(f"\nWrote {len(qualitative)} qualitative examples to {examples_path}")


if __name__ == "__main__":
    main()
