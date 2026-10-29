import argparse
import csv
import json
import logging
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import cast

import torch
from diffusers import AutoencoderKL
from PIL import Image
from torch import Tensor, nn
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import config as cfg
from generate_adversarial_image import AttackSettings, generate_adversarial_image
from models import SURROGATES, TARGETS, load_classifier


logger = logging.getLogger("vae_experiment")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LTA images and evaluate transfer.")
    parser.add_argument("--surrogate", choices=SURROGATES, default="resnet50")
    parser.add_argument(
        "--transfer-model", choices=(*TARGETS, "all"), required=True,
        help="Evaluate one paper target or all nine (including the surrogate).",
    )
    parser.add_argument("--limit", type=int, help="Process only the first N sorted images.")
    parser.add_argument("--data-dir", type=Path, default=cfg.IMAGENET_DIR)
    parser.add_argument("--output-dir", type=Path, default=cfg.OUTPUT_DIR)
    parser.add_argument("--device", default=cfg.DEVICE)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def calculate_metrics(
    total: int, clean_correct: int, adversarial_correct: int, successful_attacks: int
) -> dict[str, int | float | None]:
    if total <= 0:
        raise ValueError("Cannot calculate metrics for an empty evaluation.")
    return {
        "total": total,
        "clean_correct": clean_correct,
        "adversarial_correct": adversarial_correct,
        "successful_attacks": total - adversarial_correct,
        "asr_pct": 100 * (total - adversarial_correct) / total,
        "successful_attacks_on_clean_correct": successful_attacks,
        "clean_accuracy_pct": 100 * clean_correct / total,
        "adversarial_accuracy_pct": 100 * adversarial_correct / total,
        "adversarial_misclassification_rate_pct": 100 * (total - adversarial_correct) / total,
        "clean_correct_asr_pct": (
            100 * successful_attacks / clean_correct if clean_correct else None
        ),
    }


def evaluate_target(
    model: nn.Module,
    preprocessing: Callable[[Image.Image], Tensor],
    samples: list[tuple[Path, int]],
    image_dir: Path,
    prediction_path: Path,
    device: str,
) -> dict[str, int | float | None]:
    clean_correct = adversarial_correct = successful = 0
    with prediction_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow((
            "image", "label", "clean_prediction", "adversarial_prediction",
            "attack_success", "success_on_clean_correct",
        ))
        with torch.inference_mode():
            progress = tqdm(
                samples, desc=f"Evaluating {prediction_path.stem.removesuffix('_predictions')}",
                unit="image", dynamic_ncols=True, disable=None,
            )
            for index, (path, label) in enumerate(progress, 1):
                with Image.open(path) as source:
                    clean = source.convert("RGB").resize((cfg.W, cfg.H))
                with Image.open(image_dir / f"{path.stem}_adversarial.png") as source:
                    adversarial = source.convert("RGB")
                # Evaluate saved PNGs, with identical weight-specific transforms on both inputs.
                batch = torch.stack((preprocessing(clean), preprocessing(adversarial))).to(device)
                logits = model(batch)
                if not torch.isfinite(logits).all().item():
                    raise FloatingPointError(f"Non-finite target logits for {path.name}.")
                clean_prediction, adversarial_prediction = logits.argmax(dim=1).tolist()
                was_correct = clean_prediction == label
                is_correct = adversarial_prediction == label
                attack_success = was_correct and not is_correct
                clean_correct += int(was_correct)
                adversarial_correct += int(is_correct)
                successful += int(attack_success)
                writer.writerow((
                    path.name, label, clean_prediction, adversarial_prediction,
                    not is_correct, attack_success,
                ))
                if index == 1 or index % cfg.LOG_INTERVAL == 0 or index == len(samples):
                    logger.info("Evaluation progress | %d/%d", index, len(samples))
    return calculate_metrics(len(samples), clean_correct, adversarial_correct, successful)


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_experiment(args: argparse.Namespace, run_dir: Path) -> None:
    started = time.perf_counter()
    if cfg.LOG_INTERVAL <= 0:
        raise ValueError("LOG_INTERVAL must be positive.")
    paths = sorted(path for path in args.data_dir.glob("*_label*.png") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No ImageNet images found in {args.data_dir}")
    if args.limit is not None:
        paths = paths[:args.limit]
    samples = []
    for path in paths:
        label = int(path.stem.rsplit("_label", 1)[1])
        if not 0 <= label < 1000:
            raise ValueError(f"Invalid ImageNet-1K class ID in {path.name}")
        samples.append((path, label))
    targets = list(TARGETS) if args.transfer_model == "all" else [args.transfer_model]
    settings: AttackSettings = {
        "steps": cfg.T, "k": cfg.K, "epsilon": cfg.EPSILON,
        "penalty_weight": cfg.LAMBDA, "learning_rate": cfg.LR,
        "smoothing_interval": cfg.N, "kernel_size": cfg.SMOOTH_KERNEL_SIZE,
        "sigma": cfg.SMOOTH_SIGMA, "width": cfg.W, "height": cfg.H,
        "crop_size": cfg.CROP_SIZE, "mean": cfg.IMAGENET_MEAN,
        "std": cfg.IMAGENET_STD, "log_interval": cfg.LOG_INTERVAL,
    }
    manifest = {
        "status": "running", "surrogate": args.surrogate, "targets": targets,
        "seed": cfg.SEED, "device": args.device, "vae": cfg.VAE_MODEL,
        "vae_dtype": "float16", "latent_dtype": "float32",
        "settings": settings, "dataset": str(args.data_dir.resolve()),
        "images": [path.name for path, _ in samples],
        "evaluation": "Saved PNGs; clean images resized to attack resolution; weight-specific preprocessing for both.",
        "metrics": {
            "asr_pct": "100 * adversarial wrong / all attack attempts (untargeted).",
            "clean_correct_asr_pct": "100 * (clean correct AND adversarial wrong) / clean correct; null if no clean correct.",
            "adversarial_misclassification_rate_pct": "100 * adversarial wrong / all evaluated images.",
        },
    }
    write_json(run_dir / "run.json", manifest)
    logger.info(
        "Starting experiment | surrogate=%s | targets=%s | images=%d | seed=%d",
        args.surrogate, targets, len(samples), cfg.SEED,
    )
    logger.info("Attack settings | %s", settings)
    torch.manual_seed(cfg.SEED)
    image_dir = run_dir / "images"
    image_dir.mkdir()
    logger.info("Loading VAE | model=%s", cfg.VAE_MODEL)
    vae = AutoencoderKL.from_pretrained(cfg.VAE_MODEL, subfolder="vae", torch_dtype=torch.float16)
    cast(nn.Module, vae).to(args.device)
    logger.info("VAE ready | dtype=%s", vae.dtype)
    surrogate, _, weights = load_classifier(args.surrogate, args.device)
    manifest["surrogate_weights"] = weights
    write_json(run_dir / "run.json", manifest)
    progress = tqdm(
        samples, desc=f"Generating ({args.surrogate})",
        unit="image", dynamic_ncols=True, disable=None,
    )
    for index, (path, label) in enumerate(progress, 1):
        logger.info("Generating image %d/%d | input=%s | class_id=%d", index, len(samples), path.name, label)
        with Image.open(path) as source:
            adversarial = generate_adversarial_image(vae, surrogate, source, label, **settings)
        output_path = image_dir / f"{path.stem}_adversarial.png"
        adversarial.save(output_path)
        logger.info("Saved adversarial image | path=%s", output_path)
    del surrogate, vae
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()

    results = {}
    for name in targets:
        logger.info("Evaluating target | model=%s", name)
        model, preprocessing, weights = load_classifier(name, args.device)
        metrics = evaluate_target(
            model, preprocessing, samples, image_dir,
            run_dir / f"{name}_predictions.csv", args.device,
        )
        results[name] = {"weights": weights, "preprocessing": str(preprocessing), **metrics}
        write_json(run_dir / "results.json", results)
        logger.info("Target results | model=%s | %s", name, metrics)
        del model
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()
    manifest["status"] = "completed"
    manifest["elapsed_seconds"] = time.perf_counter() - started
    write_json(run_dir / "run.json", manifest)
    logger.info("Experiment completed | elapsed=%.1fs | results=%s", manifest["elapsed_seconds"], run_dir / "results.json")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = args.output_dir / args.surrogate / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    handlers = [logging.StreamHandler(), logging.FileHandler(run_dir / "run.log", encoding="utf-8")]
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    try:
        with logging_redirect_tqdm(loggers=[logger]):
            run_experiment(args, run_dir)
    except (Exception, KeyboardInterrupt):
        logger.exception("Experiment failed | run=%s", run_dir)
        manifest_path = run_dir / "run.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            manifest["status"] = "failed"
            write_json(manifest_path, manifest)
        raise
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()


if __name__ == "__main__":
    main()
