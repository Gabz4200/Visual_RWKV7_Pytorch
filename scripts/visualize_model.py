import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from skimage import data, segmentation

from VisualRWKV7 import spixel_upsampling, build_knn_graph, q_shift_graph_multihead, create_vision_rwkv7
from VisualRWKV7.utils.data import prepare_balanced_superpixel_features


def visualize_superpixels_and_graph(img_np, img_tensor, model, ax):
    """Visualizes diffSLIC superpixels, centroids, and the K-NN graph."""
    # Use the model's diff_slic and compactness for consistency
    diff_slic = model.diff_slic
    compactness = model.compactness

    with torch.no_grad():
        # Scale spatial channels by compactness before diffSLIC
        img_tensor_for_slic = img_tensor.clone()
        img_tensor_for_slic[:, -2:] *= compactness
        
        clst_feats, p2s_assign, _ = diff_slic(img_tensor_for_slic, n_spixels=model.num_superpixels)

    # Replicate the hard label upsampling from Vision_RWKV7.forward
    radius = diff_slic.candidate_radius
    neighbor_range = 2 * radius + 1
    h_s, w_s = clst_feats.shape[-2:]
    K = h_s * w_s

    hard_assign = (
        F.one_hot(p2s_assign.argmax(1), neighbor_range**2)
        .permute(0, 3, 1, 2)
        .contiguous()
        .float()
    )
    label_grid = torch.arange(K, dtype=torch.float, device=img_tensor.device).reshape(
        1, 1, h_s, w_s
    )

    global_labels = spixel_upsampling(label_grid, hard_assign, candidate_radius=radius)
    global_labels = global_labels.squeeze(0).squeeze(0).long().cpu().numpy()

    # 1. Draw boundaries
    img_with_boundaries = segmentation.mark_boundaries(
        img_np, global_labels, color=(1, 0, 0), mode="thick"
    )
    ax.imshow(img_with_boundaries)
    ax.set_title(f"Superpixels (K={K}) & Boundaries", fontsize=10)
    ax.axis("off")

    # 2. Calculate and draw centroids + K-NN Graph
    # Use the centroids from the model if available, or calculate from labels
    # For visualization, we need pixel coordinates, so we use the labels.
    centroids = []
    for i in range(K):
        mask = global_labels == i
        if mask.sum() > 0:
            y_coords_px, x_coords_px = np.where(mask)
            centroids.append((x_coords_px.mean(), y_coords_px.mean()))
        else:
            centroids.append((0, 0))  # Fallback

    centroids_tensor = torch.tensor(centroids, dtype=torch.float32)
    neighbors = build_knn_graph(centroids_tensor, k=4).cpu().numpy()

    # Overlay centroids and graph edges
    for i in range(K):
        cx, cy = centroids[i]
        if (
            global_labels.max() > 0 and (global_labels == i).sum() > 0
        ):  # Only draw valid superpixels
            ax.plot(cx, cy, "wo", markersize=3)  # White centroid dot
            for n_idx in neighbors[i]:
                if (global_labels == n_idx).sum() > 0:
                    nx, ny = centroids[n_idx]
                    ax.plot(
                        [cx, nx], [cy, ny], "g-", alpha=0.4, linewidth=0.8
                    )  # Green graph edge

    ax.set_title("Superpixels + K-NN Graph (k=4)", fontsize=10)


def visualize_conv_features(img_tensor, model, ax):
    """Visualizes the feature maps output by the initial Conv2d Patch Embedder."""
    # Use the full pipeline to get balanced features
    with torch.no_grad():
        balanced_features = prepare_balanced_superpixel_features(img_tensor)
        # Resize to model's expected input size for this demo
        features_input = F.interpolate(
            balanced_features, size=(224, 224), mode="bilinear", align_corners=False
        )
        # Apply the Conv layer
        features = model.patch_embed.conv(features_input)

    features = features.squeeze(0).cpu().numpy()  # [C, H', W']

    # Normalize features for visualization (0 to 1)
    features_min = features.min(axis=(1, 2), keepdims=True)
    features_max = features.max(axis=(1, 2), keepdims=True)
    features_norm = (features - features_min) / (features_max - features_min + 1e-8)

    # Plot the first 16 channels in a 4x4 grid
    num_channels = min(16, features.shape[0])
    fig_channels, axes = plt.subplots(4, 4, figsize=(8, 8))
    axes = axes.flatten()

    for i in range(16):
        if i < num_channels:
            axes[i].imshow(features_norm[i], cmap="magma")
            axes[i].set_title(f"Ch {i}", fontsize=8)
        else:
            axes[i].axis("off")
        axes[i].axis("off")

    fig_channels.suptitle(
        "Patch Embedding Conv2d Feature Maps (First 16 Channels)", fontsize=12
    )
    plt.tight_layout()
    plt.show()


def visualize_q_shift_mechanics(ax):
    """Creates a synthetic image to clearly demonstrate how Q-Shift moves pixels."""
    # Create a synthetic 8x8 image with 4 channels, each with a distinct pattern
    # Channel 0: Gradient Left-to-Right (Will shift Right)
    # Channel 1: Gradient Right-to-Left (Will shift Left)
    # Channel 2: Gradient Top-to-Bottom (Will shift Down)
    # Channel 3: Gradient Bottom-to-Top (Will shift Up)

    H, W = 8, 8
    x = np.linspace(0, 1, W)
    y = np.linspace(0, 1, H)
    X, Y = np.meshgrid(x, y)

    synthetic_img = np.zeros((1, 4, H, W), dtype=np.float32)
    synthetic_img[0, 0, :, :] = X  # Ch 0
    synthetic_img[0, 1, :, :] = 1 - X  # Ch 1
    synthetic_img[0, 2, :, :] = Y  # Ch 2
    synthetic_img[0, 3, :, :] = 1 - Y  # Ch 3

    tensor_img = torch.tensor(synthetic_img)

    # Generate grid centroids to build KNN graph for synthetic image
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    centroids_synth = torch.stack((xx.flatten(), yy.flatten()), dim=-1).float()
    neighbors_synth = build_knn_graph(centroids_synth, k=4)

    # Apply Q-Shift using the KNN graph
    shifted_img = (
        q_shift_graph_multihead(
            tensor_img.view(1, H * W, 4),
            neighbors=neighbors_synth,
            head_dim=4,
        )
        .view(1, H, W, 4)
        .numpy()[0]
    )

    original = synthetic_img[0].transpose(1, 2, 0)

    # Plot original vs shifted for Channel 0 (Right Shift) and Channel 2 (Down Shift)
    ax[0].imshow(original[:, :, 0], cmap="Blues", vmin=0, vmax=1)
    ax[0].set_title("Original (Ch 0: L→R Gradient)", fontsize=9)
    ax[0].axis("off")

    ax[1].imshow(shifted_img[:, :, 0], cmap="Blues", vmin=0, vmax=1)
    ax[1].set_title("Q-Shifted Right (Ch 0)", fontsize=9)
    ax[1].axis("off")

    ax[2].imshow(original[:, :, 2], cmap="Reds", vmin=0, vmax=1)
    ax[2].set_title("Original (Ch 2: T→B Gradient)", fontsize=9)
    ax[2].axis("off")

    ax[3].imshow(shifted_img[:, :, 2], cmap="Reds", vmin=0, vmax=1)
    ax[3].set_title("Q-Shifted Down (Ch 2)", fontsize=9)
    ax[3].axis("off")


if __name__ == "__main__":
    print("Loading sample image...")
    # Use a standard skimage image
    img_np = data.astronaut()

    # Convert to PyTorch tensor: [1, 3, H, W], normalized to [0, 1]
    img_tensor_rgb = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    
    # Preprocess to 6-channel balanced features externally
    img_tensor = prepare_balanced_superpixel_features(img_tensor_rgb)

    # Initialize a tiny model for feature extraction
    print("Initializing model...")
    model = create_vision_rwkv7(
        img_size=224, embed_dims=64, num_heads=1, depth=2, num_superpixels=150
    )

    # Create the visualization layout
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Bottom row for Q-Shift mechanics (4 panels)
    sub_gs_q = gs[1, :].subgridspec(1, 4)
    ax_q = [fig.add_subplot(sub_gs_q[0, i]) for i in range(4)]

    print("Generating Superpixel & Graph Visualization...")
    # Resize to 224x224 to match model's expected input scale
    img_resized_tensor = F.interpolate(
        img_tensor, size=(224, 224), mode="bilinear", align_corners=False
    )
    # Use RGB image for visualization background
    img_resized_rgb_tensor = F.interpolate(
        img_tensor_rgb, size=(224, 224), mode="bilinear", align_corners=False
    )
    img_resized_rgb_np = img_resized_rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    visualize_superpixels_and_graph(img_resized_rgb_np, img_resized_tensor, model, ax1)

    print("Generating Convolutional Feature Map Visualization...")
    # We'll use ax2 for a summary of conv features
    with torch.no_grad():
        features = model.patch_embed.conv(img_resized_tensor)
        features = features.squeeze(0).cpu().numpy()
    
    features_norm = (features - features.min()) / (features.max() - features.min() + 1e-8)
    ax2.imshow(features_norm[0], cmap="magma")
    ax2.set_title("Conv Feature Map (Channel 0)", fontsize=10)
    ax2.axis("off")

    print("Generating Q-Shift Mechanics Visualization...")
    visualize_q_shift_mechanics(ax_q)

    plt.suptitle(
        "Vision-RWKV-7 Internal Mechanics Visualization",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    
    # Also show the full conv feature grid in a separate window
    print("Showing full Conv feature grid...")
    visualize_conv_features(img_tensor_rgb, model, None)
    
    plt.show()
    print("Visualization complete!")
