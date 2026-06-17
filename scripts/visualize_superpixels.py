#!/usr/bin/env python3
"""
Visualize Superpixel Tokenization (diffSLIC) for Visual-RWKV-7.
"""

import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from VisualRWKV7.model import Vision_RWKV7
from VisualRWKV7.utils.data import preprocess_image_for_rwkv7
from skimage.segmentation import mark_boundaries

def main():
    parser = argparse.ArgumentParser(description="Visualize Visual-RWKV-7 Superpixel Tokenization")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("--n_count", type=int, default=196, help="Target number of superpixels")
    parser.add_argument("--compactness", type=float, default=0.5, help="Compactness factor (spatial vs color weight)")
    parser.add_argument("--iters", type=int, default=5, help="Number of diffSLIC iterations")
    parser.add_argument("--size", type=int, default=512, help="Image size for visualization")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu/cuda)")
    
    args = parser.parse_args()
    device = torch.device(args.device)

    # 1. Preprocess image for the model (L, a, b, alpha, x, y)
    # We use the full pipeline function we just created.
    x = preprocess_image_for_rwkv7(
        args.image_path,
        target_size=(args.size, args.size),
        include_alpha=True,
        chroma_scale=2.5 # Match the default used in preprocessing
    ).to(device)

    # 2. Initialize Model (Backbone)
    model = Vision_RWKV7(
        img_size=args.size,
        in_chans=6,
        embed_dims=64, # Small for visualization
        depth=1,       # Only need the first part for tokenization
        num_superpixels=args.n_count,
        diff_slic_iters=args.iters,
        compactness=args.compactness
    ).to(device).eval()

    # 3. Run Forward pass to get superpixel labels
    # We need to capture the labels. Since forward returns outs, 
    # and we want the labels, we might need to modify forward or use a hook.
    # Actually, let's just run the diff_slic part directly for visualization.
    
    with torch.no_grad():
        # Replicate the logic from Vision_RWKV7.forward
        B, _, _, _ = x.shape
        model_input = x
        
        # Apply compactness scaling to the spatial channels for diffSLIC
        slic_input = model_input.clone()
        slic_input[:, 4:6, :, :] *= args.compactness
        
        clst_feats, p2s_assign, _ = model.diff_slic(slic_input)
        h_s, w_s = clst_feats.shape[-2:]
        K = h_s * w_s
        radius = model.diff_slic.candidate_radius
        
        # Get hard labels for visualization
        import torch.nn.functional as F
        hard_assign_idx = p2s_assign.argmax(1)
        neighbor_range = 2 * radius + 1
        hard_assign_mask = F.one_hot(hard_assign_idx, neighbor_range**2).permute(0, 3, 1, 2).float()
        
        label_grid = (
            torch.arange(K, dtype=torch.float, device=x.device)
            .reshape(1, 1, h_s, w_s)
            .expand(B, -1, -1, -1)
        )
        
        from VisualRWKV7.utils.diffSLIC_funcs import spixel_upsampling
        global_labels = (
            spixel_upsampling(label_grid, hard_assign_mask, candidate_radius=radius)
            .squeeze(1)
            .long()
        )
        
    # 4. Prepare image for display (Convert back to sRGB)
    # Load raw image again for display to avoid normalization artifacts if possible, 
    # or just invert the normalization.
    # Let's load it raw and resize.
    display_img = Image.open(args.image_path).convert("RGB").resize((args.size, args.size))
    display_np = np.array(display_img)
    
    labels_np = global_labels[0].cpu().numpy()
    
    # 5. Visualize
    fig, ax = plt.subplots(figsize=(10, 10))
    # Use mark_boundaries from skimage
    out = mark_boundaries(display_np, labels_np, color=(1, 1, 0)) # Yellow boundaries
    
    ax.imshow(out)
    ax.set_title(f"Visual-RWKV-7 Superpixels (n_count={args.n_count}, compactness={args.compactness})")
    ax.axis("off")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
