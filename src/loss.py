

from typing import Sequence

import torch
import torch.nn as nn

from utils import apply_perturbation


def margin_loss(
    logits: torch.Tensor, label: torch.Tensor, kappa: float = 0.0
) -> torch.Tensor:
    """Untargeted Carlini-Wagner margin, written in maximize-me form.

    The CW objective ``max(z_y - max_{i != y} z_i, -kappa)`` is minimized, so
    this returns its negation, ``min(max_{i != y} z_i - z_y, kappa)``, which the
    caller maximizes exactly like the cross-entropy term. The clamp stops the
    attack once the strongest competitor leads by ``kappa`` logits, spending the
    remaining budget on images that are not yet misclassified.
    """
    if logits.ndim != 2:
        raise ValueError("Expected logits with shape (B, num_classes).")
    if logits.shape[1] < 2:
        raise ValueError("Margin loss needs at least two classes.")
    if kappa < 0:
        raise ValueError("kappa must be non-negative.")

    true_logit = logits.gather(1, label.view(-1, 1)).squeeze(1)
    mask = nn.functional.one_hot(label, logits.shape[1]).bool()
    best_other = logits.masked_fill(mask, float("-inf")).amax(dim=1)
    return torch.clamp(best_other - true_logit, max=kappa).mean()


def eot_loss(
    image: torch.Tensor,
    label: torch.Tensor,
    surrogate_model: nn.Module,
    k: int,
    crop_size: int,
    mean: Sequence[float],
    std: Sequence[float],
    objective: str = "margin",
    kappa: float = 0.0,
) -> torch.Tensor:
    """EOT surrogate objective averaged over k perturbations of a [0, 1] image.

    Returns a quantity the caller maximizes: the negated CW margin when
    ``objective`` is ``"margin"``, or cross-entropy when it is ``"ce"``.
    ImageNet normalization is applied only on the classifier input path.
    """
    if objective not in ("ce", "margin"):
        raise ValueError(f"Unsupported objective: {objective!r}")
    criterion = nn.CrossEntropyLoss()
    mean_t = torch.tensor(mean, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    total = torch.zeros((), device=image.device, dtype=torch.float32)
    for _ in range(k):
        perturbed = apply_perturbation(image, crop_size)
        perturbed = (perturbed - mean_t) / std_t
        logits = surrogate_model(perturbed)
        total = total + (
            criterion(logits, label)
            if objective == "ce"
            else margin_loss(logits, label, kappa)
        )
    return total / k

# Change to L2 ? for smoother. 
def soft_penalty(
    adversarial_image: torch.Tensor, original_image: torch.Tensor, epsilon: float
) -> torch.Tensor:
    """Mean per-pixel amount by which |adv - orig| exceeds epsilon."""
    excess = torch.abs(adversarial_image - original_image) - epsilon
    return nn.functional.relu(excess).sum() / adversarial_image.numel()
