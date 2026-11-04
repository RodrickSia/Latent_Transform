

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

def soft_penalty(
    adversarial_image: torch.Tensor | None = None,
    original_image: torch.Tensor | None = None,
    epsilon: float = 0.0,
    latent_mode: bool = False,
    norm: str = "l2",
    adversarial_latent_vector: torch.Tensor | None = None,
    original_latent_vector: torch.Tensor | None = None,
) -> torch.Tensor:
    """Penalty on the difference between adversarial and original inputs.

    Pixel mode (default) compares ``adversarial_image`` and ``original_image``.
    With ``latent_mode=True`` it compares ``adversarial_latent_vector`` and
    ``original_latent_vector`` instead; the image arguments are ignored.

    ``"l1"`` returns the mean absolute difference (the paper's form).
    ``"l2"`` returns the L2 norm of the difference; it is roughly ``sqrt(N)``
    times larger than the L1 mean, so ``penalty_weight`` must be retuned when
    switching.

    ``"none"`` disables the penalty (returns zero).
    """
    if norm not in ("l1", "l2", "none"):
        raise ValueError(f"Unsupported penalty norm: {norm!r}")
    if norm == "none":
        return torch.zeros((), device=(adversarial_image if adversarial_image is not None else adversarial_latent_vector).device)
    if latent_mode:
        if adversarial_latent_vector is None or original_latent_vector is None:
            raise ValueError(
                "latent_mode requires adversarial_latent_vector and original_latent_vector."
            )
        adversarial, original = adversarial_latent_vector, original_latent_vector
    else:
        if adversarial_image is None or original_image is None:
            raise ValueError("Pixel mode requires adversarial_image and original_image.")
        adversarial, original = adversarial_image, original_image

    difference = adversarial - original
    if norm == "l1":
        return torch.abs(difference).sum() / difference.numel()
    return torch.sqrt(torch.square(difference).sum() + 1e-12)  # Adding this to prevent gradient exploding at zero
