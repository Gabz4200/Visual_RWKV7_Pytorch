"""Complete test and debug of diffSLIC to ensure it works on any image."""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from VisualRWKV7.diffSLIC import DiffSLIC, spixel_upsampling
import os


def test_diffslic_on_multiple_images():
    """Tests diffSLIC on different types of images."""

    # diffSLIC settings
    n_spixels = 196  # 14x14
    n_iter = 10
    tau = 0.01
    candidate_radius = 1

    diff_slic = DiffSLIC(
        n_spixels=n_spixels,
        n_iter=n_iter,
        tau=tau,
        candidate_radius=candidate_radius,
        normalize=True,
        stable=False,
    )

    # List of test images
    test_images = []

    # 1. Synthetic image - gradient
    grad_img = torch.zeros(1, 3, 224, 224)
    for c in range(3):
        grad_img[0, c, :, :] = torch.linspace(0, 1, 224).unsqueeze(0).repeat(224, 1) * (
            c + 1
        )
    test_images.append(("Gradient", grad_img))

    # 2. Synthetic image - checkerboard
    checker = torch.zeros(1, 3, 224, 224)
    block_size = 16
    for i in range(0, 224, block_size):
        for j in range(0, 224, block_size):
            color = ((i // block_size) + (j // block_size)) % 2
            checker[0, :, i : i + block_size, j : j + block_size] = color
    test_images.append(("Checkerboard", checker))

    # 3. Random image (noisy)
    noise_img = torch.randn(1, 3, 224, 224) * 0.5
    test_images.append(("Random Noise", noise_img))

    # 4. Image with uniform colors
    uniform_img = torch.ones(1, 3, 224, 224) * 0.5
    uniform_img[0, :, :112, :] = 0.2  # Dark half
    uniform_img[0, :, 112:, :] = 0.8  # Light half
    test_images.append(("Uniform Halves", uniform_img))

    # 5. Real image (if available)
    try:
        real_img = Image.open("test_image.jpg").convert("RGB")
        real_img = (
            torch.from_numpy(np.array(real_img)).permute(2, 0, 1).unsqueeze(0).float()
            / 255.0
        )
        real_img = F.interpolate(real_img, size=(224, 224), mode="bilinear")
        test_images.append(("Real Image", real_img))
    except:
        print("No real image found, skipping...")

    print(f"Testing diffSLIC with {n_spixels} superpixels, {n_iter} iterations\n")
    print("=" * 80)

    results = []

    for img_name, img_tensor in test_images:
        print(f"\nTesting: {img_name}")
        print(f"  Shape: {img_tensor.shape}")
        print(f"  Range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")

        try:
            with torch.no_grad():
                # Run diffSLIC
                clst_feats, p2s_assign, s2p_assign = diff_slic(img_tensor)

                # Get superpixel dimensions
                h_s, w_s = clst_feats.shape[-2:]
                K = h_s * w_s

                print(f"  Generated superpixels: {K} ({h_s}x{w_s})")

                # Convert to labels
                radius = diff_slic.candidate_radius
                neighbor_range = 2 * radius + 1
                hard_assign = (
                    F.one_hot(p2s_assign.argmax(1), neighbor_range**2)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                    .float()
                )

                label_grid = torch.arange(
                    K, dtype=torch.float, device=img_tensor.device
                ).reshape(1, 1, h_s, w_s)

                global_labels = (
                    spixel_upsampling(label_grid, hard_assign, candidate_radius=radius)
                    .squeeze(1)
                    .long()
                )

                # Superpixel statistics
                labels_np = global_labels[0].cpu().numpy()
                unique_labels, counts = np.unique(labels_np, return_counts=True)

                print(f"  Unique labels: {len(unique_labels)}")
                print(f"  Average size: {counts.mean():.1f} pixels")
                print(f"  Min size: {counts.min()} pixels")
                print(f"  Max size: {counts.max()} pixels")

                # Check for empty superpixels
                expected_labels = set(range(K))
                actual_labels = set(unique_labels.flatten())
                missing_labels = expected_labels - actual_labels

                if len(missing_labels) > 0:
                    print(f"  WARNING: {len(missing_labels)} empty superpixels!")
                else:
                    print(f"  [OK] All superpixels are present")

                results.append(
                    {
                        "name": img_name,
                        "img": img_tensor,
                        "labels": global_labels,
                        "K": K,
                        "unique": len(unique_labels),
                        "missing": len(missing_labels),
                    }
                )

        except Exception as e:
            print(f"  ERROR: {str(e)}")
            import traceback

            traceback.print_exc()
            results.append({"name": img_name, "error": str(e)})

    print("\n" + "=" * 80)
    print("TEST SUMMARY:")
    print("=" * 80)

    for r in results:
        if "error" in r:
            print(f"[FAIL] {r['name']}: ERROR - {r['error']}")
        else:
            status = "[OK]" if r["missing"] == 0 else "[WARN]"
            print(
                f"{status} {r['name']}: {r['unique']}/{r['K']} superpixels, {r['missing']} missing"
            )

    # Visualization
    fig, axes = plt.subplots(2, len(results), figsize=(4 * len(results), 8))
    if len(results) == 1:
        axes = axes.reshape(-1, 1)

    for i, r in enumerate(results):
        if "error" not in r:
            # Original image
            img_np = r["img"][0].permute(1, 2, 0).cpu().numpy()
            img_np = np.clip(img_np, 0, 1)
            axes[0, i].imshow(img_np)
            axes[0, i].set_title(f"{r['name']}\n(Original)")
            axes[0, i].axis("off")

            # Superpixel labels
            labels_color = label_to_rgb(r["labels"][0].cpu().numpy())
            axes[1, i].imshow(labels_color)
            axes[1, i].set_title(f"{r['name']}\n(Superpixels)")
            axes[1, i].axis("off")
        else:
            axes[0, i].text(
                0.5,
                0.5,
                f"ERROR:\n{r['error']}",
                ha="center",
                va="center",
                transform=axes[0, i].transAxes,
            )
            axes[0, i].axis("off")
            axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig("diffslic_test_results.png", dpi=150, bbox_inches="tight")
    print("\nVisualization saved to: diffslic_test_results.png")
    plt.show()

    return results


def label_to_rgb(labels):
    """Converts integer labels to colored RGB."""
    from matplotlib.colors import hsv_to_rgb

    h, w = labels.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    # Unique colors for each label
    unique_labels = np.unique(labels)
    n_colors = len(unique_labels)

    # Generate evenly spaced HSV colors
    hues = np.linspace(0, 1, n_colors, endpoint=False)
    saturation = np.ones(n_colors) * 0.8
    value = np.ones(n_colors) * 0.9

    hsv_colors = np.stack([hues, saturation, value], axis=1)
    rgb_colors = hsv_to_rgb(hsv_colors)

    # Map labels to colors
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

    for i in range(h):
        for j in range(w):
            label = labels[i, j]
            idx = label_to_idx[label]
            rgb[i, j] = rgb_colors[idx]

    return rgb


def test_with_different_configs():
    """Tests different hyperparameter configurations."""
    print("\n" + "=" * 80)
    print("TESTING DIFFERENT CONFIGURATIONS")
    print("=" * 80)

    # Default test image
    test_img = torch.randn(1, 3, 224, 224) * 0.5

    configs = [
        {"n_spixels": 49, "n_iter": 5, "tau": 0.01},  # Few superpixels
        {"n_spixels": 196, "n_iter": 10, "tau": 0.01},  # Default config
        {"n_spixels": 400, "n_iter": 15, "tau": 0.005},  # Many superpixels
        {"n_spixels": 196, "n_iter": 20, "tau": 0.001},  # Many iterations
    ]

    for i, config in enumerate(configs):
        print(f"\nConfig {i + 1}: {config}")

        diff_slic = DiffSLIC(
            n_spixels=config["n_spixels"],
            n_iter=config["n_iter"],
            tau=config["tau"],
            candidate_radius=1,
            normalize=True,
        )

        try:
            with torch.no_grad():
                clst_feats, p2s_assign, _ = diff_slic(test_img)
                h_s, w_s = clst_feats.shape[-2:]
                K_actual = h_s * w_s
                print(f"  [OK] Success: {K_actual} superpixels generated")
        except Exception as e:
            print(f"  [FAIL] Failure: {str(e)}")


if __name__ == "__main__":
    print("STARTING COMPLETE DIFFSLIC TESTS")
    print("=" * 80)

    # Test 1: Multiple images
    results = test_diffslic_on_multiple_images()

    # Test 2: Different configurations
    test_with_different_configs()

    # Final summary
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    all_passed = all("error" not in r and r["missing"] == 0 for r in results)

    if all_passed:
        print("[OK] ALL TESTS PASSED!")
        print("diffSLIC is working correctly on all images.")
    else:
        print("[WARN] SOME TESTS FAILED")
        print("Check the errors above and adjust the parameters.")
        print("\nSuggestions:")
        print("  - Increase n_iter for more iterations (10-20)")
        print("  - Adjust tau (0.001-0.1) for smoothness control")
        print("  - Check if the image has enough contrast")
        print("  - Reduce n_spixels if there are too many empty superpixels")
