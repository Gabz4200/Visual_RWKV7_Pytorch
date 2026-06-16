"""Vision-RWKV-7 demo: verify backbone with dummy image input."""

import torch
from VisualRWKV7.model import Vision_RWKV7


def main():
    torch.manual_seed(42)

    # Initialize the new Vision-RWKV-7 with Superpixel Tokenization (diffSLIC)
    model = Vision_RWKV7(
        img_size=224,
        in_chans=3,
        embed_dims=192,
        num_heads=3,
        depth=12,
        init_values=1e-5,
        final_norm=True,
        out_indices=[3, 5, 7, 11],
        num_superpixels=196,  # Target number of superpixels (approx 14x14 grid)
        diff_slic_iters=5,  # Number of iterations for diffSLIC optimization
    )

    # Dummy image input
    x = torch.randn(2, 3, 224, 224)

    # Forward pass
    outs = model(x)

    print(f"Input:  {tuple(x.shape)}")
    print(f"Output levels:  {len(outs)}")
    for i, o in enumerate(outs):
        print(f"  level {i}: {tuple(o.shape)}")

    total = sum(p.numel() for p in model.parameters())
    print(f"\nTotal params: {total / 1e6:.2f}M")

    # Verify no NaN/Inf
    all_finite = all(o.isfinite().all() for o in outs)
    print(f"All outputs finite: {all_finite}")

    # Verify deterministic: same input -> same output
    outs2 = model(x)
    deterministic = all(
        (o1 - o2).abs().max().item() < 1e-5 for o1, o2 in zip(outs, outs2)
    )
    print(f"Deterministic: {deterministic}")


if __name__ == "__main__":
    main()
