#let ans(body) = {
  block(
    width: 100%,
    stroke: 1pt + black,
    inset: 10pt,
    radius: 5pt,
    fill: luma(250),
    [
      *Solution*: #body
    ]
  )
}
#let prf(body) = { [_Proof:_ #body $square$] }
#let qs(body) = {
  set enum(numbering: "(a)")
  body
}
#let pt(body) = {
  body
}

#align(center)[
  = CS 148b -- HW 3
  Marco Yang
]

== 2 ViT Design

=== 2.4 Pooling

+ #qs[
  We chose to use the [CLS] token as the image summary. An alternative is to mean-pool all patch embeddings after the final block, or to use an attention-pooling head. For downstream tasks that require spatial reasoning (e.g., counting objects, OCR, or visual question answering that refers to specific image regions), which pooling strategy do you expect to perform best, and why? What information is lost when we condense an entire image into a single CLS vector before passing it to a language model?
  
  #ans[
    I think they're all roughly equivalent in terms of representation power, with the average pooling being slightly worse since it doesn't use the positional relationship at all and averages over all patches. Attention pooling seems like the cleanest approach since we can take a petrained model and just learn one extra embedding and use it to average the final embeddings properly. The CLS approach is theoretically more powerful since we are doing attention pooling over every layer and also using that same attention-pooled representation as an extra token. Regardless, I think that compressing the entire image into one CLS vector destroys the spatial information since the size of our representation doesn't scale with the amount of information in the image (image size/resolution).
  ]
]

=== 2.4 Effect of patch size

Consider a 224 $times$ 224 RGB image and patch sizes $P in {8,16,32}$.

+ #qs[
  Compute the number of patches $N$ for each $P$. What happens to the self-attention compute cost (which scales as $O(N^2 d_"model")$) as you shrink $P$?

  #ans[]
]

+ #qs[
  Measure forward-pass wall-clock time on a single batch of 16 images for each $P$ using a ViT with `d_model = 384`, `num_heads = 6`, `num_blocks = 6`. Use `torch.cuda.synchronize()` around your timing block, and average over 20 steps after 5 warmup steps.

  #ans[]
]

+ #qs[
  Smaller patches preserve more spatial detail but are more expensive. In one sentence, when would you accept this trade-off?

  #ans[]
]

== 3. CLIP-Style Contrastive Pretraining

=== 3.2 Symmetric InfoNCE Loss

+ #qs[
  Include a 1--2 sentence explanation of why the CLIP loss is symmetric (i.e., averaged in both directions).

  #ans[]
]

=== 3.3 CLIP Pretraining on EuroSAT

+ #qs[
  Report (a) a training-loss curve, (b) a zero-shot validation accuracy curve, and (c) 2--3 sentences on how the two curves relate. Does training loss continue to improve after validation accuracy plateaus?

  #ans[]
]

=== 3.3 Zero-Shot Qualitative Analysis

+ #qs[
  Pick 5 correctly classified and 5 incorrectly classified validation images. For each incorrectly classified image, inspect the top-3 predicted classes. Are the mistakes "reasonable" (e.g., PermanentCrop mistaken for HerbaceousVegetation) or nonsensical? What does this tell you about the structure of the learned embedding space?

  #ans[]
]

== 4. LoRA Fine-Tuning

=== 4.1 LoRA Parameter Count

+ #qs[
  Include a printout showing (i) total parameters, (ii) trainable parameters, and (iii) the ratio, for your ViT with LoRA rank 8.

  #ans[]
]

=== 4.2 Full FT vs. LoRA vs. Linear Probe

+ #qs[
  Starting from your CLIP-pretrained ViT, compare linear probe, LoRA (rank 8, $alpha = 16$), and full fine-tuning on RESISC45. For each method, report (a) final test accuracy, (b) number of trainable parameters, (c) peak GPU memory during training, and (d) wall-clock training time. Discuss the trade-offs in 4--5 sentences.

  #ans[]
]

=== 4.2 LoRA Rank Sweep

+ #qs[
  Sweep the LoRA rank $r in {1, 2, 4, 8, 16, 32, 64}$ with $alpha = 2r$. Plot test accuracy as a function of rank. (1) At what rank do you see diminishing returns? (2) How does your answer compare to the rank at which LoRA is typically deployed in practice (e.g., $r = 8$ or $r = 16$ in large-model fine-tuning)? What does this tell you about the effective rank of the fine-tuning update?

  #ans[]
]

== 5. Vision-Language Model

=== 5.3 Vision-Language Projector

+ #qs[
  Include a 1--2 sentence rationale for why we need more than a single linear layer in the vision-language projector. Hint: think about what additional learnable capacity buys you when the encoder and decoder are kept frozen during the pretraining stage of VLM training.

  #ans[]
]

=== 5.4 Injection Strategy Comparison

+ #qs[
  Train a VLM with each of the three injection strategies (CLS-only prefix, all-patches prefix, and interleaved via placeholder) for 2000 steps on CLEVR. For each strategy, report (1) validation exact-match accuracy on 500 held-out CLEVR examples, (2) number of visual tokens injected per example, (3) peak GPU memory during training, and (4) wall-clock time per step. Which strategy gives the best accuracy, and is the extra cost worth it? You should observe a clear connection to the CLS-vs-patch pooling question from §2.4.

  #ans[]
]

=== 5.5 Attention Masking

+ #qs[
  Draw the attention mask for a sequence of 4 visual tokens followed by 3 text tokens, under each of (M1) fully causal and (M2) bidirectional inside image / causal across boundary. Use a $7 times 7$ grid with shaded cells for allowed positions.

  #ans[]
]

+ #qs[
  Which of (M1) and (M2) do you expect to perform better, and why?

  #ans[]
]

+ #qs[
  Train with each mask for 500 steps on CLEVR (using the all-patches injection strategy) and report validation accuracy.

  #ans[]
]

=== 5.6 Freezing Strategies

+ #qs[
  Starting from the best injection + masking configuration, run four training configurations for 1500 steps each:
  - *A (projector only)*: encoder frozen, projector trained, decoder frozen
  - *B (projector + decoder LoRA)*: encoder frozen, projector trained, decoder LoRA (rank 8)
  - *C (projector + full decoder)*: encoder frozen, projector trained, decoder full FT
  - *D (all three)*: encoder full FT, projector full FT, decoder full FT

  Report validation exact-match accuracy, trainable parameter count, and peak memory for each. Which configuration gives the best trade-off between accuracy and cost? Discuss in the context of the two-stage (pretraining, instruction-tuning) recipe.

  #ans[]
]

=== 5.7 Qualitative Evaluation

+ #qs[
  Take your best VLM and generate responses on 10 held-out CLEVR examples. Include the image, the question, the ground-truth answer, and your model's generation. Pick a mix of correct and incorrect cases. For each incorrect case, hypothesize whether the failure is in the encoder (image not well understood) or the decoder (language component misinterpreting the question). How would you design an experiment to distinguish between these two failure modes?

  #ans[]
]

== 6. Positional Encodings and RoPE

=== 6.1 1D RoPE

+ #qs[
  Verify manually that applying RoPE preserves the norm of each vector (up to numerical precision), and report what you measure.

  #ans[]
]

=== 6.1 Learned PE vs. RoPE in the ViT

+ #qs[
  Retrain CLIP-style on EuroSAT for 20 epochs using (a) learned PE and (b) 1D RoPE. Report zero-shot validation accuracy for each. Then evaluate both models on EuroSAT images upsampled to $96 times 96$ (keeping patch size 8), which produces 144 patches instead of the 64 seen at training. For the learned-PE baseline, interpolate the learned patch positional embeddings from the $8 times 8$ training grid to the $12 times 12$ evaluation grid. How does each model's accuracy degrade?

  #ans[]
]

=== 6.2 2D RoPE for Image Patches

+ #qs[
  Swap RoPE1D for RoPE2D in your ViT (using each patch's $(x, y)$ grid indices) and re-run the CLIP pretraining + zero-shot evaluation. Does 2D RoPE improve over 1D RoPE on EuroSAT? Include the length-extrapolation test with 2D RoPE as well. Discuss in 2--3 sentences.

  #ans[]
]

=== 6.3 Multimodal RoPE (M-RoPE)

+ #qs[
  What goes wrong with naive 1D position IDs $(0, 1, 2, dots)$ when we inject 64 patch tokens plus a CLS token before a 50-token text prompt? Think about (a) position-ID values the decoder was trained on, and (b) the 2D structure of the image.

  #ans[]
]

+ #qs[
  Under M-RoPE, what position does the first text token get (as a function of the image's grid size)? Why is this choice sensible?

  #ans[]
]

+ #qs[
  Why does M-RoPE split the head dimension into three chunks rather than two? What would break if we only used $(x, y)$ and dropped the temporal $t$?

  #ans[]
]

=== 6.3 Implementing M-RoPE (Bonus)

+ #qs[
  Implement an M-RoPE-style position assignment for your VLM. Retrain the best configuration from §5 for 1500 steps, using (a) naive 1D position IDs and (b) M-RoPE-style position IDs. Does M-RoPE improve CLEVR accuracy? Does it help more on questions that refer to spatial relations ("left of", "behind", "in front of") than on other questions?

  #ans[]
]
