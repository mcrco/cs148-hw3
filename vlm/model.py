"""Vision-Language Model — §5.

You implement: VisionLanguageModel.

Three injection strategies to support:
  - "cls":          Single visual token (the ViT's CLS embedding) prepended.
  - "all_patches":  All N+1 visual tokens (CLS + patches) prepended.
  - "interleaved":  A special <image> token in the prompt is replaced by the
                    sequence of patch embeddings at runtime.

Two attention masking strategies to support (Problem `masking`):
  - "causal":         Fully causal across the whole sequence.
  - "image_bidir":    Bidirectional within the image block, causal everywhere
                      else. Use vlm.masking.build_image_bidir_mask().
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from .masking import build_causal_mask, build_image_bidir_mask

InjectionMode = Literal["cls", "all_patches", "interleaved"]
MaskMode = Literal["causal", "image_bidir"]


class VisionLanguageModel(nn.Module):
    """ViT image encoder + projector + pretrained causal LM decoder.

    Args:
        vit:       Your CLIP-pretrained ViT from §3.
        projector: vlm.projector.VisionLanguageProjector instance.
        decoder:   HuggingFace causal LM (e.g., SmolLM2-360M-Instruct) loaded
                   in bf16 with FlashAttention-2.
        tokenizer: Matching HF tokenizer.
        image_token_id: Token ID corresponding to the special <image> placeholder
                        in interleaved mode (None for cls / all_patches modes).

    Forward:
        images:         (B, 3, H, W) float tensor.
        input_ids:      (B, T) tokenized text.
        attention_mask: (B, T) text attention mask from the tokenizer.
        labels:         (B, T) for loss computation, or None for inference.
                        Visual-token positions must be set to -100 in labels
                        before being passed in (so they're masked out by HF's
                        loss).
        injection:      One of "cls", "all_patches", "interleaved".
        mask_mode:      One of "causal", "image_bidir".

    Returns:
        A dict with at least:
          - "loss":   scalar (only if labels was provided).
          - "logits": (B, T_total, vocab_size).
    """

    def __init__(
        self,
        vit: nn.Module,
        projector: nn.Module,
        decoder: nn.Module,
        tokenizer,
        image_token_id: int | None = None,
    ) -> None:
        super().__init__()
        self.vit = vit
        self.projector = projector
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.image_token_id = image_token_id

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        injection: InjectionMode = "cls",
        mask_mode: MaskMode = "causal",
    ) -> dict:
        image_features = self.vit(
            images, return_all_tokens=injection in ("all_patches", "interleaved")
        )
        projector_dtype = next(self.projector.parameters()).dtype
        image_features = image_features.to(dtype=projector_dtype)
        image_embeds = self.projector(image_features)
        text_embeds = self.decoder.get_input_embeddings()(input_ids)
        image_embeds = image_embeds.to(dtype=text_embeds.dtype)

        B = text_embeds.shape[0]
        num_image_tokens = image_embeds.shape[1]

        if injection == "cls" or injection == "all_patches":
            token_embeds = torch.cat((image_embeds, text_embeds), dim=1)
            padding_attention_mask = torch.cat(
                (
                    torch.ones(
                        B,
                        num_image_tokens,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                    attention_mask,
                ),
                dim=1,
            )
            if labels is not None:
                labels = torch.cat(
                    (
                        torch.full(
                            (B, num_image_tokens),
                            -100,
                            dtype=labels.dtype,
                            device=labels.device,
                        ),
                        labels,
                    ),
                    dim=1,
                )

            if mask_mode == "image_bidir":
                decoder_attention_mask = build_image_bidir_mask(
                    num_image_tokens,
                    text_embeds.shape[1],
                    device=token_embeds.device,
                    dtype=token_embeds.dtype,
                ).expand(B, -1, -1, -1)
            else:
                decoder_attention_mask = padding_attention_mask
        elif injection == "interleaved":
            if self.image_token_id is None:
                raise ValueError("image_token_id must be set for interleaved injection")

            is_image = input_ids == self.image_token_id
            if not torch.all(is_image.sum(dim=1) == 1):
                raise ValueError("interleaved injection expects exactly one <image> token per row")

            image_indices = is_image.int().argmax(dim=1)
            token_embeds_list = []
            attention_mask_list = []
            labels_list = [] if labels is not None else None
            for i in range(B):
                image_token_idx = int(image_indices[i].item())
                token_embeds_list.append(
                    torch.cat(
                        (
                            text_embeds[i, :image_token_idx],
                            image_embeds[i],
                            text_embeds[i, image_token_idx + 1 :],
                        ),
                        dim=0,
                    )
                )
                attention_mask_list.append(
                    torch.cat(
                        (
                            attention_mask[i, :image_token_idx],
                            torch.ones(
                                num_image_tokens,
                                dtype=attention_mask.dtype,
                                device=attention_mask.device,
                            ),
                            attention_mask[i, image_token_idx + 1 :],
                        ),
                        dim=0,
                    )
                )
                if labels is not None:
                    labels_list.append(
                        torch.cat(
                            (
                                labels[i, :image_token_idx],
                                torch.full(
                                    (num_image_tokens,),
                                    -100,
                                    dtype=labels.dtype,
                                    device=labels.device,
                                ),
                                labels[i, image_token_idx + 1 :],
                            ),
                            dim=0,
                        )
                    )

            token_embeds = torch.stack(token_embeds_list, dim=0)
            padding_attention_mask = torch.stack(attention_mask_list, dim=0)
            if labels_list is not None:
                labels = torch.stack(labels_list, dim=0)

            if mask_mode == "image_bidir":
                seq_len = token_embeds.shape[1]
                decoder_attention_mask = (
                    build_causal_mask(seq_len, device=token_embeds.device, dtype=token_embeds.dtype)
                    .expand(B, -1, -1, -1)
                    .clone()
                )
                for i in range(B):
                    start = int(image_indices[i].item())
                    end = start + num_image_tokens
                    decoder_attention_mask[i, :, start:end, start:end] = 0
            else:
                decoder_attention_mask = padding_attention_mask

        if mask_mode == "image_bidir":
            key_padding_mask = torch.zeros(
                (B, 1, 1, token_embeds.shape[1]),
                device=token_embeds.device,
                dtype=token_embeds.dtype,
            )
            key_padding_mask = key_padding_mask.masked_fill(
                padding_attention_mask[:, None, None, :].to(device=token_embeds.device) == 0,
                torch.finfo(token_embeds.dtype).min,
            )
            decoder_attention_mask = decoder_attention_mask + key_padding_mask

        output = self.decoder(
            inputs_embeds=token_embeds,
            attention_mask=decoder_attention_mask,
            labels=labels,
        )

        result = {"logits": output.logits}
        if labels is not None:
            result["loss"] = output.loss
        return result

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        prompts: list[str],
        injection: InjectionMode = "cls",
        mask_mode: MaskMode = "causal",
        max_new_tokens: int = 32,
        **gen_kwargs,
    ) -> list[str]:
        """Generate text continuations conditioned on images + prompts.

        Useful for §5's qualitative evaluation problem (vlm_qualitative).
        """
        was_training = self.training
        self.eval()

        tokenizer_kwargs = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "add_special_tokens": False,
        }
        encoded = self.tokenizer(prompts, **tokenizer_kwargs)
        input_ids = encoded["input_ids"].to(images.device)
        attention_mask = encoded["attention_mask"].to(images.device)

        eos_token_id = gen_kwargs.pop("eos_token_id", self.tokenizer.eos_token_id)
        pad_token_id = gen_kwargs.pop("pad_token_id", self.tokenizer.pad_token_id)
        do_sample = bool(gen_kwargs.pop("do_sample", False))
        temperature = float(gen_kwargs.pop("temperature", 1.0))
        top_p = float(gen_kwargs.pop("top_p", 1.0))

        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=images.device)
        generated: list[torch.Tensor] = []

        for _ in range(max_new_tokens):
            out = self.forward(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,
                injection=injection,
                mask_mode=mask_mode,
            )
            logits = out["logits"][:, -1, :]
            if do_sample:
                logits = logits / max(temperature, 1e-6)
                probs = torch.softmax(logits, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                    cumulative = torch.cumsum(sorted_probs, dim=-1)
                    remove = cumulative > top_p
                    remove[:, 0] = False
                    sorted_probs = sorted_probs.masked_fill(remove, 0.0)
                    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                    sampled = torch.multinomial(sorted_probs, num_samples=1)
                    next_token = sorted_idx.gather(-1, sampled)
                else:
                    next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            if eos_token_id is not None:
                fill_id = pad_token_id if pad_token_id is not None else eos_token_id
                next_token = torch.where(
                    finished[:, None],
                    torch.full_like(next_token, fill_id),
                    next_token,
                )
                finished |= next_token.squeeze(1) == eos_token_id

            generated.append(next_token)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
            if bool(finished.all()):
                break

        if was_training:
            self.train()

        if not generated:
            return ["" for _ in prompts]
        generated_ids = torch.cat(generated, dim=1)
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
