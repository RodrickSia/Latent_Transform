from typing import Sequence

import torch
import torch.nn as nn

from utils import apply_perturbation


def eot_loss(
    image: torch.Tensor,
    label: torch.Tensor,
    surrogate_model: nn.Module,
    k: int,
    crop_size: int,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    """EOT cross-entropy averaged over k perturbations of a [0, 1] image.

    ImageNet normalization is applied only on the classifier input path.
    """
    criterion = nn.CrossEntropyLoss()
    mean_t = torch.tensor(mean, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    total = torch.zeros((), device=image.device, dtype=torch.float32)
    for _ in range(k):
        perturbed = apply_perturbation(image, crop_size)
        perturbed = (perturbed - mean_t) / std_t
        total = total + criterion(surrogate_model(perturbed), label)
    return total / k


def soft_penalty(
    adversarial_image: torch.Tensor, original_image: torch.Tensor, epsilon: float
) -> torch.Tensor:
    """Mean per-pixel amount by which |adv - orig| exceeds epsilon."""
    excess = torch.abs(adversarial_image - original_image) - epsilon
    return nn.functional.relu(excess).sum() / adversarial_image.numel()
