import logging
import time
from collections.abc import Sequence
from typing import TypedDict, cast

import torch
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
from torch import nn
from tqdm.auto import tqdm

from loss import eot_loss, soft_penalty
from utils import depthwise_smoothing


logger = logging.getLogger("vae_experiment.attack")


class AttackSettings(TypedDict):
    steps: int
    k: int
    epsilon: float
    penalty_weight: float
    learning_rate: float
    smoothing_interval: int
    kernel_size: int
    sigma: float
    width: int
    height: int
    crop_size: int
    mean: Sequence[float]
    std: Sequence[float]
    log_interval: int


def generate_adversarial_image(
    vae: AutoencoderKL,
    surrogate_model: nn.Module,
    image: Image.Image,
    label: int,
    *,
    steps: int,
    k: int,
    epsilon: float,
    penalty_weight: float,
    learning_rate: float,
    smoothing_interval: int,
    kernel_size: int,
    sigma: float,
    width: int,
    height: int,
    crop_size: int,
    mean: Sequence[float],
    std: Sequence[float],
    log_interval: int,
) -> Image.Image:
    """Attack one PIL image and return its final RGB image.

    Both models must already be on the same device. They are frozen and placed
    in evaluation mode. The caller controls seeding, logging, and saving.
    """
    if min(steps, k, smoothing_interval, log_interval) <= 0:
        raise ValueError("Steps, EOT samples, smoothing interval, and log interval must be positive.")
    if not 0 <= label < 1000:
        raise ValueError("Label must be an ImageNet class index in [0, 999].")
    vae.eval().requires_grad_(False)
    surrogate_model.eval().requires_grad_(False)
    device = next(vae.parameters()).device
    processor = VaeImageProcessor(vae_scale_factor=8)
    resized = image.convert("RGB").resize((width, height))
    x = processor.preprocess(resized).to(device=device, dtype=vae.dtype)
    target = torch.tensor([label], device=device, dtype=torch.long)
    started = time.perf_counter()
    logger.info("Encoding input | class_id=%d", label)
    with torch.no_grad():
        z_0 = vae.encode(x, return_dict=False)[0].mode().float()
    z = z_0.clone().requires_grad_(True)
    logger.info("Latent ready | shape=%s | dtype=%s", tuple(z.shape), z.dtype)
    optimizer = torch.optim.Adam([z], lr=learning_rate)
    original = (x.float() / 2 + 0.5).clamp(0, 1)

    progress = tqdm(
        range(1, steps + 1), desc="Optimizing latent", unit="step",
        leave=False, dynamic_ncols=True, disable=None,
    )
    for step in progress:
        decoded = vae.decode(z.to(dtype=vae.dtype), return_dict=False)[0]  # pyright: ignore[reportArgumentType]
        # Algorithm 1 evaluates both loss terms on the raw decoder output; the
        # soft penalty, not a clamp, is what keeps pixels inside [0, 1].
        adversarial = decoded.float() / 2 + 0.5
        classification_loss = eot_loss(
            adversarial, target, surrogate_model, k, crop_size, mean, std
        )
        penalty = soft_penalty(adversarial, original, epsilon)
        loss = -classification_loss + penalty_weight * penalty
        if not torch.isfinite(loss).item():
            logger.error("Non-finite loss at step %d/%d", step, steps)
            raise FloatingPointError(f"Non-finite loss at step {step}/{steps}.")
        if step == 1 or step % log_interval == 0 or step == steps:
            progress.set_postfix(
                loss=f"{loss.item():.4f}", penalty=f"{penalty.item():.4f}"
            )
            logger.info(
                "Step %d/%d (pre-update) | loss=%.6f | eot_ce=%.6f | "
                "penalty=%.6f | pixel_linf=%.6f | elapsed=%.1fs",
                step, steps, loss.item(), classification_loss.item(), penalty.item(),
                (adversarial.detach() - original).abs().max().item(),
                time.perf_counter() - started,
            )
        optimizer.zero_grad()
        loss.backward()
        if z.grad is None or not torch.isfinite(z.grad).all().item():
            raise FloatingPointError(f"Missing or non-finite latent gradient at step {step}.")
        optimizer.step()
        if not torch.isfinite(z).all().item():
            raise FloatingPointError(f"Non-finite latent after step {step}.")
        if step % smoothing_interval == 0:
            with torch.no_grad():
                delta = depthwise_smoothing(z - z_0, kernel_size, sigma)
                z.copy_(z_0 + delta)
            logger.info("Applied latent smoothing after step %d", step)

    logger.info("Decoding final latent")
    with torch.no_grad():
        decoded = vae.decode(z.to(dtype=vae.dtype), return_dict=False)[0]  # pyright: ignore[reportArgumentType]
        if not torch.isfinite(decoded).all().item():
            raise FloatingPointError("Non-finite final decoded image.")
        return cast(
            list[Image.Image], processor.postprocess(decoded.float(), output_type="pil")
        )[0]
