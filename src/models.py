import logging
from collections.abc import Callable

from PIL import Image
from torch import Tensor, nn
from torchvision.models import get_model, get_model_weights


SURROGATES = ("resnet50", "resnet152", "vgg16")
TARGETS = (
    "resnet50", "resnet152", "vgg16", "mobilenet_v2", "inception_v3",
    "vit_b_16", "pit_b", "visformer_small", "swin_t",
)
logger = logging.getLogger("vae_experiment.models")


def load_classifier(
    name: str, device: str
) -> tuple[nn.Module, Callable[[Image.Image], Tensor], str]:
    """Load a paper architecture and its pretrained evaluation preprocessing."""
    if name not in TARGETS:
        raise ValueError(f"Unsupported classifier: {name}")
    logger.info("Loading classifier | model=%s | device=%s", name, device)
    if name in ("pit_b", "visformer_small"):
        import timm
        from timm.data import create_transform, resolve_model_data_config

        model_name = "pit_b_224" if name == "pit_b" else name
        model = timm.create_model(model_name, pretrained=True)
        preprocessing = create_transform(
            **resolve_model_data_config(model), is_training=False
        )
        weights_name = model.pretrained_cfg["hf_hub_id"]
    else:
        weights = get_model_weights(name).DEFAULT
        model = get_model(name, weights=weights)
        preprocessing = weights.transforms()
        weights_name = str(weights)
    model.to(device).eval().requires_grad_(False)
    logger.info("Classifier ready | model=%s | weights=%s", name, weights_name)
    return model, preprocessing, weights_name
