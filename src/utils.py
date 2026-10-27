import math

import torch
import torch.nn as nn


def get_gaussian_kernel(kernel_size: int, sigma: float, channels: int) -> torch.Tensor:
    """Return normalized Gaussian weights with shape (channels, 1, size, size)."""
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer.")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive.")
    if channels <= 0:
        raise ValueError("channels must be positive.")

    coordinates = torch.arange(kernel_size, dtype=torch.float32)
    coordinates = coordinates - (kernel_size - 1) / 2
    gaussian = torch.exp(-0.5 * (coordinates / sigma).square())
    kernel = torch.outer(gaussian, gaussian)
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)


def depthwise_smoothing(
    latent_var: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0
) -> torch.Tensor:
    """Smooth each NCHW channel independently, preserving shape and gradients.

    The Gaussian kernel matches the input device and dtype. Zero padding keeps
    the spatial dimensions unchanged.
    """
    if latent_var.ndim != 4:
        raise ValueError("Expected a latent tensor with shape (B, C, H, W).")
    if not latent_var.is_floating_point():
        raise TypeError("latent_var must be a floating-point tensor.")

    channels = latent_var.shape[1]
    kernel = get_gaussian_kernel(kernel_size, sigma, channels).to(
        device=latent_var.device, dtype=latent_var.dtype
    )
    return nn.functional.conv2d(
        latent_var, kernel, padding=kernel_size // 2, groups=channels
    )


def apply_perturbation(image: torch.Tensor, size: int = 224) -> torch.Tensor:
    """Uniformly choose one of three differentiable perturbations per call."""
    if image.ndim not in (3, 4):
        raise ValueError("Expected a CHW or NCHW image tensor.")
    unbatched = image.ndim == 3
    batch = image.unsqueeze(0) if unbatched else image

    perturbation = int(torch.randint(3, ()).item())
    if perturbation == 0:
        # Random resize: integer size sampled uniformly from [size, size + 64].
        resize_size = int(torch.randint(size, size + 65, ()).item())
        perturbed = nn.functional.interpolate(
            batch,
            size=(resize_size, resize_size),
            mode="bilinear",
            align_corners=False,
        )
    elif perturbation == 1:
        # Random interpolation mode at a fixed target size.
        modes = ("nearest", "bilinear", "bicubic")
        mode = modes[int(torch.randint(len(modes), ()).item())]
        perturbed = nn.functional.interpolate(
            batch,
            size=(size, size),
            mode=mode,
            align_corners=False if mode in ("bilinear", "bicubic") else None,
        )
    else:
        # Crop near the center, jittering each axis by up to 8px.
        height, width = batch.shape[-2:]
        if height < size or width < size:
            raise ValueError(f"Cropping requires height and width >= {size}.")
        center_top = (height - size) // 2
        center_left = (width - size) // 2
        top = int(torch.randint(
            max(0, center_top - 8), min(height - size, center_top + 8) + 1, ()
        ).item())
        left = int(torch.randint(
            max(0, center_left - 8), min(width - size, center_left + 8) + 1, ()
        ).item())
        perturbed = batch[..., top:top + size, left:left + size]

    return perturbed.squeeze(0) if unbatched else perturbed
