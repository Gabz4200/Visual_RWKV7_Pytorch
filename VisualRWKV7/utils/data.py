import torch
import math
from PIL import Image
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from typing import Optional, Tuple, cast

# Import color conversion utilities from your provided colors.py
from .colors import from_srgb_to_linear_rgb, from_linear_rgb_to_oklab

# ==============================================================================
# Default Normalization Constants
# ==============================================================================

# ImageNet RGB statistics.
# These were calculated by the original authors over the entire ImageNet
# training set after scaling pixel values to [0.0, 1.0].
# Added a neutral placeholder for Alpha channel (Mean=0.5, Std=0.5).
IMAGENET_RGB_MEAN = [0.485, 0.456, 0.406, 0.5]
IMAGENET_RGB_STD = [0.229, 0.224, 0.225, 0.5]

# Heuristic OkLAB statistics for natural images.
# DEPRECATED: Use Fixed Balancing (2.0 * L - 1.0) instead of dataset stats.
# These are kept only for legacy compatibility.
DEFAULT_OKLAB_MEAN = [0.5, 0.0, 0.0, 0.5]
DEFAULT_OKLAB_STD = [0.2, 0.15, 0.15, 0.5]


def _convert_srgb_to_oklab(srgb_tensor: torch.Tensor) -> torch.Tensor:
    """
    Helper to convert an sRGB tensor [0, 1] to OkLAB.
    Uses the exact pipeline: sRGB -> Linear RGB -> OkLAB.
    Supports 3 (RGB) or 4 (RGBA) channels.
    """
    linear_rgb = from_srgb_to_linear_rgb(srgb_tensor)
    oklab = from_linear_rgb_to_oklab(linear_rgb)
    return oklab


def calculate_dataset_mean_std(
    data_dir: str,
    img_size: int = 224,
    batch_size: int = 64,
    color_space: str = "rgb",
    include_alpha: bool = False,
):
    """
    Calculates the mean and std of a dataset in either RGB or OkLAB space.

    Args:
        data_dir (str): Path to the dataset folder.
        img_size (int): Target image size.
        batch_size (int): Batch size for DataLoader.
        color_space (str): 'rgb' or 'oklab'.
        include_alpha (bool): Whether to include the alpha channel in calculations.
    """
    # 1. Transform to tensor ONLY. Do NOT normalize yet!
    # ToTensor automatically converts PIL images to sRGB tensors in [0.0, 1.0]
    # Note: ToTensor handles RGBA images by producing 4 channels.
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.Lambda(
                lambda x: x.convert("RGBA") if include_alpha else x.convert("RGB")
            ),
            transforms.ToTensor(),
        ]
    )

    # 2. Load the dataset
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 3. Initialize accumulators
    num_channels = 4 if include_alpha else 3
    sum_pixels = torch.zeros(num_channels)
    sum_sq_pixels = torch.zeros(num_channels)
    total_pixels = 0

    print(
        f"Calculating mean and std in {color_space.upper()} space... (This may take a few minutes)"
    )

    # 4. Iterate through the dataset
    for images, _ in loader:
        # If OkLAB is requested, convert the sRGB batch to OkLAB before calculating stats
        if color_space == "oklab":
            images = _convert_srgb_to_oklab(images)

        # images shape: (Batch, Channels, Height, Width)
        # Flatten H and W to calculate stats per channel
        _, c, _, _ = images.shape
        pixels = images.permute(0, 2, 3, 1).reshape(-1, c)  # Shape: (B*H*W, C)

        # Accumulate sums
        sum_pixels += pixels.sum(dim=0)
        sum_sq_pixels += (pixels**2).sum(dim=0)
        total_pixels += pixels.shape[0]

    # 5. Calculate final mean and std
    mean = sum_pixels / total_pixels
    var = (sum_sq_pixels / total_pixels) - mean**2
    std = torch.sqrt(torch.clamp(var, min=0.0))

    return mean.tolist(), std.tolist()


def load_image_to_tensor(
    image_path: str,
    target_size: Optional[Tuple[int, int]] = None,
    normalize: bool = False,
    color_space: str = "rgb",
    mean: Optional[list] = None,
    std: Optional[list] = None,
    include_alpha: bool = False,
):
    """
    Loads an image, optionally converts to OkLAB, and normalizes it.

    Args:
        image_path (str): Path to the image file.
        target_size (Tuple[int, int], optional): Target (Height, Width).
        normalize (bool): If True, normalizes the tensor.
        color_space (str): 'rgb' (default) or 'oklab'.
        mean (list, optional): Mean for normalization. If None, uses defaults.
        std (list, optional): Std for normalization. If None, uses defaults.
        include_alpha (bool): If True, includes the alpha channel (RGBA).

    Returns:
        torch.Tensor: Tensor of shape (1, 3 or 4, Height, Width).
    """

    # 1. Open image and convert to RGBA if alpha is requested, else RGB
    img = Image.open(image_path)
    if include_alpha:
        img = img.convert("RGBA")
    else:
        img = img.convert("RGB")

    # 2. Build the transformation pipeline
    transform_list = []
    if target_size is not None:
        transform_list.append(transforms.Resize(target_size))

    # ToTensor converts PIL Image (H, W, C) in [0, 255] to sRGB Tensor (C, H, W) in [0.0, 1.0]
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)

    # 3. Apply transforms (Resulting shape: C, H, W)
    img_tensor = cast(torch.Tensor, transform(img))

    # 4. Add the Batch dimension at the front (Resulting shape: B, C, H, W where B=1)
    img_tensor = img_tensor.unsqueeze(0)

    # 5. Convert to OkLAB if requested
    if color_space == "oklab":
        img_tensor = _convert_srgb_to_oklab(img_tensor)
    elif color_space != "rgb":
        raise ValueError(
            f"Unsupported color_space: '{color_space}'. Use 'rgb' or 'oklab'."
        )

    # 6. Normalize if requested
    if normalize:
        # Determine which default constants to use based on the color space
        if color_space == "rgb":
            norm_mean = (
                mean
                if mean is not None
                else (IMAGENET_RGB_MEAN if include_alpha else IMAGENET_RGB_MEAN[:3])
            )
            norm_std = (
                std
                if std is not None
                else (IMAGENET_RGB_STD if include_alpha else IMAGENET_RGB_STD[:3])
            )
        else:
            # OkLAB normalization is deprecated in favor of Fixed Balancing.
            # If still requested, we use the legacy defaults.
            norm_mean = (
                mean
                if mean is not None
                else (DEFAULT_OKLAB_MEAN if include_alpha else DEFAULT_OKLAB_MEAN[:3])
            )
            norm_std = (
                std
                if std is not None
                else (DEFAULT_OKLAB_STD if include_alpha else DEFAULT_OKLAB_STD[:3])
            )

        # Apply normalization manually using tensor broadcasting.
        # Equivalent to transforms.Normalize(mean=norm_mean, std=norm_std) but more explicit.
        # mean and std are reshaped to (1, C, 1, 1) to broadcast across (B, C, H, W)
        mean_t = torch.tensor(norm_mean, dtype=img_tensor.dtype).view(1, -1, 1, 1)
        std_t = torch.tensor(norm_std, dtype=img_tensor.dtype).view(1, -1, 1, 1)
        img_tensor = (img_tensor - mean_t) / std_t

    return img_tensor


def add_spatial_coordinates(
    tensor: torch.Tensor, center_origin: bool = True
) -> torch.Tensor:
    """
    Appends spatial coordinate channels (x, y) to the input tensor.

    Args:
        tensor (torch.Tensor): Input tensor of shape (B, C, H, W).
        center_origin (bool): If True, maps coordinates to [-1, 1] with the
                              exact center of the image at (0, 0).
                              If False, maps to [0, 1] with top-left at (0, 0).

    Returns:
        torch.Tensor: Tensor of shape (B, C+2, H, W).
                      The new channels are x (width/cols) and y (height/rows).
    """
    B, _, H, W = tensor.shape

    if not tensor.is_floating_point():
        tensor = tensor.float()

    dtype = tensor.dtype
    device = tensor.device

    # 1. Generate 1D coordinate tensors
    if center_origin:
        # linspace(-1, 1, N) perfectly centers the grid at 0.
        # For odd N (e.g., 5): [-1.0, -0.5, 0.0, 0.5, 1.0] -> Exact 0 in the middle.
        # For even N (e.g., 4): [-1.0, -0.333, 0.333, 1.0] -> 0 falls exactly between the two center pixels.
        x_coords = torch.linspace(-1.0, 1.0, W, dtype=dtype, device=device)
        y_coords = torch.linspace(-1.0, 1.0, H, dtype=dtype, device=device)
    else:
        # Standard [0, 1] mapping
        x_coords = torch.linspace(0.0, 1.0, W, dtype=dtype, device=device)
        y_coords = torch.linspace(0.0, 1.0, H, dtype=dtype, device=device)

    # 2. Create 2D grids
    # indexing='ij' ensures grid_y varies along the row dimension (H)
    # and grid_x varies along the column dimension (W)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

    # 3. Expand to match the batch dimension (memory-free view)
    grid_x = grid_x.view(1, 1, H, W).expand(B, -1, -1, -1)
    grid_y = grid_y.view(1, 1, H, W).expand(B, -1, -1, -1)

    # 4. Concatenate along the channel dimension
    # Order: original channels, then x, then y
    return torch.cat([tensor, grid_x, grid_y], dim=1)


def smart_resize(
    height: int,
    width: int,
    spixel_size: int = 16,
    min_pixels: int = 4 * 16 * 16,
    max_pixels: int = 1024 * 1024,
) -> Tuple[int, int]:
    """
    Calculates a 'smart' resolution for an image, similar to Qwen2-VL.
    Ensures height and width are multiples of spixel_size and total pixels
    are within [min_pixels, max_pixels] while preserving aspect ratio.
    """
    if height < spixel_size or width < spixel_size:
        # Scale up if too small
        scale = max(spixel_size / height, spixel_size / width)
        height, width = int(height * scale), int(width * scale)

    # Clamp total pixels
    pixels = height * width
    if pixels < min_pixels:
        scale = math.sqrt(min_pixels / pixels)
        height, width = int(height * scale), int(width * scale)
    elif pixels > max_pixels:
        scale = math.sqrt(max_pixels / pixels)
        height, width = int(height * scale), int(width * scale)

    # Ensure multiples of spixel_size
    new_h = max(spixel_size, round(height / spixel_size) * spixel_size)
    new_w = max(spixel_size, round(width / spixel_size) * spixel_size)

    return new_h, new_w


def preprocess_image_for_rwkv7(
    image_path: str,
    target_size: Optional[Tuple[int, int]] = None,
    spixel_size: Optional[int] = 16,
    include_alpha: bool = True,
    chroma_scale: float = 2.5,
) -> torch.Tensor:
    """
    Full pipeline: Load -> Smart Resize -> Convert to OkLAB -> Fixed Balancing -> Add Coordinates.
    Returns a balanced tensor of shape (1, 6, H, W).

    Args:
        image_path (str): Path to the image file.
        target_size (Tuple[int, int], optional): Explicit (Height, Width). 
                                                 If None and spixel_size is set, uses smart_resize.
        spixel_size (int, optional): Superpixel size for smart_resize.
        include_alpha (bool): If True, includes the alpha channel (RGBA).
        chroma_scale (float): Multiplier for 'a' and 'b' channels.
    """
    img = Image.open(image_path)
    w, h = img.size

    if target_size is None and spixel_size is not None:
        target_size = smart_resize(h, w, spixel_size=spixel_size)

    # 1. Load image as sRGB tensor [0, 1]
    x_srgb = load_image_to_tensor(
        image_path,
        target_size=target_size,
        normalize=False,
        color_space="rgb",
        include_alpha=include_alpha,
    )

    # 2. Split RGB and Alpha
    if include_alpha:
        srgb = x_srgb[:, :3, :, :]
        alpha = x_srgb[:, 3:4, :, :]
    else:
        srgb = x_srgb
        alpha = None

    # 3. Apply Fixed Balancing (Conversion + Scaling + Coordinates)
    # This replaces the need for dataset-specific mean/std.
    return prepare_balanced_superpixel_features(
        srgb, alpha=alpha, chroma_scale=chroma_scale
    )


def prepare_balanced_superpixel_features(
    srgb_image: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    chroma_scale: float = 2.5,
) -> torch.Tensor:
    """
    End-to-end preprocessing from a standard sRGB image to balanced 6-channel
    superpixel features ready for diffSLIC.

    Args:
        srgb_image (torch.Tensor): Input image of shape (B, 3, H, W) in [0, 1].
        alpha (torch.Tensor, optional): Transparency mask of shape (B, 1, H, W) in [0, 1].
                                        If None, assumes fully opaque (1.0).
        chroma_scale (float): Multiplier for 'a' and 'b' channels to match the
                              [-1, 1] magnitude of the other channels.

    Returns:
        torch.Tensor: Balanced tensor of shape (B, 6, H, W) containing
                      [L_bal, a_bal, b_bal, alpha_bal, x, y].
    """
    B, C, H, W = srgb_image.shape
    assert C == 3, f"Input image must have 3 channels (RGB), got {C}"

    dtype = srgb_image.dtype
    device = srgb_image.device

    # =========================================================================
    # 1. Color Space Conversion (sRGB -> Linear RGB -> OkLAB)
    # =========================================================================
    linear_rgb = from_srgb_to_linear_rgb(srgb_image)
    oklab = from_linear_rgb_to_oklab(linear_rgb)

    L = oklab[:, 0:1, :, :]  # Shape: (B, 1, H, W)
    a = oklab[:, 1:2, :, :]  # Shape: (B, 1, H, W)
    b = oklab[:, 2:3, :, :]  # Shape: (B, 1, H, W)

    # =========================================================================
    # 2. Handle Alpha Channel
    # =========================================================================
    if alpha is None:
        # Default to fully opaque if no alpha mask is provided
        alpha = torch.ones(B, 1, H, W, dtype=dtype, device=device)

    # =========================================================================
    # 3. Generate Spatial Coordinates (x, y) centered at (0, 0)
    # =========================================================================
    # Maps coordinates to [-1, 1] with the exact center of the image at (0, 0)
    y_coords = torch.linspace(-1.0, 1.0, H, dtype=dtype, device=device)
    x_coords = torch.linspace(-1.0, 1.0, W, dtype=dtype, device=device)

    # Create 2D grids (indexing='ij' ensures y varies along rows, x along cols)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

    # Expand to batch dimension (memory-free view, no extra allocation yet)
    x = grid_x.view(1, 1, H, W).expand(B, -1, -1, -1)
    y = grid_y.view(1, 1, H, W).expand(B, -1, -1, -1)

    # =========================================================================
    # 4. Apply Balancing / Normalization (The core math)
    # =========================================================================
    # Shift L and Alpha from [0, 1] to [-1, 1] to center them at 0
    L_bal = 2.0 * L - 1.0
    alpha_bal = 2.0 * alpha - 1.0

    # Scale 'a' and 'b' to match the [-1, 1] magnitude of the spatial coords
    # (OkLAB 'a' and 'b' for sRGB rarely exceed [-0.4, 0.4], so 2.5 stretches them nicely)
    a_bal = a * chroma_scale
    b_bal = b * chroma_scale

    # x and y are already perfectly in [-1, 1] with mean 0.

    # =========================================================================
    # 5. Concatenate into final 6-channel tensor
    # =========================================================================
    # Order: [L, a, b, alpha, x, y]
    balanced_features = torch.cat([L_bal, a_bal, b_bal, alpha_bal, x, y], dim=1)

    return balanced_features


def revert_balanced_superpixel_features(
    balanced_features: torch.Tensor,
    chroma_scale: float = 2.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Inverse operation of prepare_balanced_superpixel_features.
    Converts balanced 6-channel features back to valid OkLAB and Alpha.

    Args:
        balanced_features (torch.Tensor): Balanced tensor of shape (B, 6, H, W)
                                          [L_bal, a_bal, b_bal, alpha_bal, x, y].
        chroma_scale (float): Multiplier used during the forward balancing step.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - oklab: Tensor of shape (B, 3, H, W) [L, a, b] in valid OkLAB range.
            - alpha: Tensor of shape (B, 1, H, W) [alpha] in [0, 1].
    """
    # 1. Extract channels
    L_bal = balanced_features[:, 0:1, :, :]
    a_bal = balanced_features[:, 1:2, :, :]
    b_bal = balanced_features[:, 2:3, :, :]
    alpha_bal = balanced_features[:, 3:4, :, :]

    # 2. Invert Balancing Math
    # L and Alpha: [-1, 1] -> [0, 1]
    # original = (balanced + 1.0) / 2.0
    L = (L_bal + 1.0) / 2.0
    alpha = (alpha_bal + 1.0) / 2.0

    # a and b: [-1, 1] -> native OkLAB range
    # original = balanced / chroma_scale
    a = a_bal / chroma_scale
    b = b_bal / chroma_scale

    # 3. Concatenate OkLAB channels
    oklab = torch.cat([L, a, b], dim=1)

    return oklab, alpha
