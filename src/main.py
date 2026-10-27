import logging
import time
from typing import cast

import torch
import torch.nn as nn
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50

import config as cfg
from loss import eot_loss, soft_penalty
from utils import depthwise_smoothing


logger = logging.getLogger(__name__)


def main() -> None:
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = cfg.OUTPUT_DIR / "attack.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    if cfg.LOG_INTERVAL <= 0:
        raise ValueError("LOG_INTERVAL must be positive.")
    started = time.perf_counter()
    logger.info("Starting attack | device=%s | seed=%d | log=%s", cfg.DEVICE, cfg.SEED, log_path)
    logger.info(
        "Settings | steps=%d | lr=%g | K=%d | epsilon=%.6f | lambda=%g | size=%dx%d",
        cfg.T, cfg.LR, cfg.K, cfg.EPSILON, cfg.LAMBDA, cfg.W, cfg.H,
    )
    logger.info(
        "Smoothing | interval=%d | kernel=%d | sigma=%g",
        cfg.N, cfg.SMOOTH_KERNEL_SIZE, cfg.SMOOTH_SIGMA,
    )
    torch.manual_seed(cfg.SEED)
    # The downloader saves each sample as <image_id>_label<class_id>.png.
    
    # Get image and label
    image_paths = sorted(
        path for path in cfg.IMAGENET_DIR.glob("*_label*.png") if path.is_file()
    )
    if not image_paths:
        raise FileNotFoundError(f"No ImageNet images found in {cfg.IMAGENET_DIR}")
    if not 0 <= cfg.IMAGE_INDEX < len(image_paths):
        raise IndexError(f"IMAGE_INDEX must be between 0 and {len(image_paths) - 1}.")

    image_path = image_paths[cfg.IMAGE_INDEX]
    class_id = int(image_path.stem.rsplit("_label", 1)[1])
    if not 0 <= class_id < 1000:
        raise ValueError(f"Invalid ImageNet-1K class ID in {image_path.name}")
    logger.info("Input | path=%s | class_id=%d", image_path, class_id)

    # Init VAE and surrogate models
    logger.info("Loading VAE | model=%s", cfg.VAE_MODEL)
    vae = AutoencoderKL.from_pretrained(
        cfg.VAE_MODEL, subfolder="vae", torch_dtype=torch.float16
    )
    cast(nn.Module, vae).to(cfg.DEVICE)
    vae.requires_grad_(False)
    logger.info("VAE ready | dtype=%s", vae.dtype)

    logger.info("Loading surrogate | model=ResNet50 | weights=DEFAULT")
    surrogate = resnet50(weights=ResNet50_Weights.DEFAULT).to(cfg.DEVICE).eval()
    surrogate.requires_grad_(False)
    logger.info("Surrogate ready")

    proc = VaeImageProcessor(vae_scale_factor=8)
    with Image.open(image_path) as source:
        img = source.convert("RGB").resize((cfg.W, cfg.H))
    x = proc.preprocess(img).to(cfg.DEVICE, torch.float16)  # [-1, 1]
    label = torch.tensor([class_id], device=cfg.DEVICE, dtype=torch.long)
    
    # Encode to latent space
    logger.info("Encoding input")
    with torch.no_grad():
        z_0 = vae.encode(x, return_dict=False)[0].sample().float()

    z = z_0.clone().requires_grad_(True)
    logger.info("Latent ready | shape=%s | dtype=%s", tuple(z.shape), z.dtype)
    
    optimizer = torch.optim.Adam([z], lr=cfg.LR)

    x_01 = (x.float() / 2 + 0.5).clamp(0, 1)  # pixel space [0, 1]
    # Optimize in latent space
    logger.info("Starting optimization")
    for t in range(1, cfg.T + 1):
        # Diffusers' FloatTensor annotation also accepts FP16 CUDA tensors at runtime.
        out = vae.decode(z.to(dtype=vae.dtype), return_dict=False)[0]  # pyright: ignore[reportArgumentType]

        adv = (out.float() / 2 + 0.5).clamp(0, 1)
        classification_loss = eot_loss(
            adv,
            label,
            surrogate,
            k=cfg.K,
            crop_size=cfg.CROP_SIZE,
            mean=cfg.IMAGENET_MEAN,
            std=cfg.IMAGENET_STD,
        )
        penalty = soft_penalty(adv, x_01, epsilon=cfg.EPSILON)
        loss = -classification_loss + cfg.LAMBDA * penalty
        if not torch.isfinite(loss).item():
            logger.error("Non-finite loss at step %d/%d", t, cfg.T)
            raise FloatingPointError(f"Non-finite loss at step {t}/{cfg.T}.")
        if t == 1 or t % cfg.LOG_INTERVAL == 0 or t == cfg.T:
            logger.info(
                "Step %d/%d (pre-update) | loss=%.6f | eot_ce=%.6f | "
                "penalty=%.6f | pixel_linf=%.6f | elapsed=%.1fs",
                t, cfg.T, loss.item(), classification_loss.item(), penalty.item(),
                (adv.detach() - x_01).abs().max().item(), time.perf_counter() - started,
            )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t % cfg.N == 0:
            with torch.no_grad():
                delta = depthwise_smoothing(
                    z - z_0, cfg.SMOOTH_KERNEL_SIZE, cfg.SMOOTH_SIGMA
                )
                z.copy_(z_0 + delta)
            logger.info("Applied latent smoothing after step %d", t)
    # Decode the final adversarial latent to image space
    logger.info("Decoding final latent")
    with torch.no_grad():
        out = vae.decode(z.to(dtype=vae.dtype), return_dict=False)[0]  # pyright: ignore[reportArgumentType]
        adversarial_image = cast(
            list[Image.Image], proc.postprocess(out.float(), output_type="pil")
        )[0]

    output_path = cfg.OUTPUT_DIR / f"{image_path.stem}_adversarial.png"
    adversarial_image.save(output_path)
    logger.info(
        "Saved adversarial image | path=%s | total_elapsed=%.1fs",
        output_path, time.perf_counter() - started,
    )


if __name__ == "__main__":
    main()
