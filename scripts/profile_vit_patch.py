import statistics
import time

import torch

import basics

if __name__ == "__main__":
    img_size = 224
    patch_sizes = [8, 16, 32]
    d_model = 384
    num_heads = 6
    num_blocks = 6
    batch_size = 16
    warmup_steps = 5
    steps = 20
    for ps in patch_sizes:
        vit = basics.vit.ViT(img_size, ps, d_model, num_heads, num_blocks)
        x = torch.randn(batch_size, 3, img_size, img_size)

        for _ in range(warmup_steps):
            _ = vit(x)

        times = []
        for _ in range(steps):
            start_time = time.perf_counter()
            _ = vit(x)
            torch.cuda.synchronize()
            duration = time.perf_counter() - start_time
            times.append(duration)

        avg = statistics.mean(times)
        stdev = statistics.stdev(times)
        print(f"Average forward-pass time for size {ps}: {avg}.")
        print(f"Stdev of forward-pass time for size {ps}: {stdev}.")
