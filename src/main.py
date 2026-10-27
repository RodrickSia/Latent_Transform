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


def main() -> None:
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

    # Init VAE and surrogate models
    vae = AutoencoderKL.from_pretrained(
        cfg.VAE_MODEL, subfolder="vae", torch_dtype=torch.float16
    )
    cast(nn.Module, vae).to(cfg.DEVICE)
    vae.requires_grad_(False)

    surrogate = resnet50(weights=ResNet50_Weights.DEFAULT).to(cfg.DEVICE).eval()
    surrogate.requires_grad_(False)

    proc = VaeImageProcessor(vae_scale_factor=8)
    with Image.open(image_path) as source:
        img = source.convert("RGB").resize((cfg.W, cfg.H))
    x = proc.preprocess(img).to(cfg.DEVICE, torch.float16)  # [-1, 1]
    label = torch.tensor([class_id], device=cfg.DEVICE, dtype=torch.long)
    
    # Encode to latent space
    with torch.no_grad():
        z_0 = vae.encode(x, return_dict=False)[0].sample().float()

    z = z_0.clone().requires_grad_(True)
    
    optimizer = torch.optim.Adam([z], lr=cfg.LR)

    x_01 = (x.float() / 2 + 0.5).clamp(0, 1)  # pixel space [0, 1]
    # Optimize in latent space
    for t in range(1, cfg.T + 1):
        # Diffusers' FloatTensor annotation also accepts FP16 CUDA tensors at runtime.
        out = vae.decode(z.to(dtype=vae.dtype), return_dict=False)[0]  # pyright: ignore[reportArgumentType]

        adv = (out.float() / 2 + 0.5).clamp(0, 1)
        loss = -eot_loss(
            adv,
            label,
            surrogate,
            k=cfg.K,
            crop_size=cfg.CROP_SIZE,
            mean=cfg.IMAGENET_MEAN,
            std=cfg.IMAGENET_STD,
        ) + cfg.LAMBDA * soft_penalty(adv, x_01, epsilon=cfg.EPSILON)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t % cfg.N == 0:
            with torch.no_grad():
                delta = depthwise_smoothing(
                    z - z_0, cfg.SMOOTH_KERNEL_SIZE, cfg.SMOOTH_SIGMA
                )
                z.copy_(z_0 + delta)
    # Decode the final adversarial latent to image space
    with torch.no_grad():
        out = vae.decode(z.to(dtype=vae.dtype), return_dict=False)[0]  # pyright: ignore[reportArgumentType]
        adversarial_image = cast(
            list[Image.Image], proc.postprocess(out.float(), output_type="pil")
        )[0]

    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = cfg.OUTPUT_DIR / f"{image_path.stem}_adversarial.png"
    adversarial_image.save(output_path)
    print(f"Saved adversarial image to {output_path}")


if __name__ == "__main__":
    main()
