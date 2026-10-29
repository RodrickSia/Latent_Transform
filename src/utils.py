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


def apply_perturbation(
    image: torch.Tensor, size: int = 224, max_scale: int = 64, jitter: int = 8
) -> torch.Tensor:
    """Sample one EOT transform and apply all of its stages to the image.

    A single transform composes a random resize to ``[size, size + max_scale]``
    using a randomly chosen interpolation kernel, followed by a near-center crop
    back to ``size`` with up to ``jitter`` pixels of offset per axis. Every stage
    runs on every call; only their parameters are resampled, so the output is
    always ``size`` x ``size`` and remains differentiable w.r.t. the input.
    """
    if image.ndim not in (3, 4):
        raise ValueError("Expected a CHW or NCHW image tensor.")
    if size <= 0 or max_scale < 0 or jitter < 0:
        raise ValueError("size must be positive; max_scale and jitter must be non-negative.")
    unbatched = image.ndim == 3
    batch = image.unsqueeze(0) if unbatched else image

    resize_size = int(torch.randint(size, size + max_scale + 1, ()).item())
    modes = ("nearest", "bilinear", "bicubic")
    mode = modes[int(torch.randint(len(modes), ()).item())]
    resized = nn.functional.interpolate(
        batch,
        size=(resize_size, resize_size),
        mode=mode,
        align_corners=False if mode in ("bilinear", "bicubic") else None,
    )

    # Jitter collapses to a single offset when resize_size == size.
    center = (resize_size - size) // 2
    low = max(0, center - jitter)
    high = min(resize_size - size, center + jitter)
    top = int(torch.randint(low, high + 1, ()).item())
    left = int(torch.randint(low, high + 1, ()).item())
    perturbed = resized[..., top:top + size, left:left + size]

    return perturbed.squeeze(0) if unbatched else perturbed
