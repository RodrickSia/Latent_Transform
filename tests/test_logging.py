import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image
from torch import nn
from torchvision.transforms import Compose, Resize, ToTensor

import config as cfg
import main
from generate_adversarial_image import generate_adversarial_image
from loss import soft_penalty
from utils import apply_perturbation


class TinyVAE(nn.Module):
    dtype = torch.float32

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def encode(self, image, return_dict=False):
        latent = nn.functional.avg_pool2d(image.float(), 8)
        return (SimpleNamespace(sample=lambda: latent, mode=lambda: latent),)

    def decode(self, latent, return_dict=False):
        return (nn.functional.interpolate(latent, size=(32, 32)).tanh(),)


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.images = self.path / "images"
        self.images.mkdir()
        for index in range(2):
            Image.new("RGB", (32, 32), (100, 150, 200)).save(
                self.images / f"{index:05d}_label2.png"
            )
        self.output = self.path / "output"
        config_patch = patch.multiple(
            cfg, LOG_INTERVAL=2, T=3, N=2, K=1, H=32, W=32, CROP_SIZE=16,
        )
        config_patch.start()
        self.addCleanup(config_patch.stop)

    def run_main(self, *extra):
        def load_classifier(name, device):
            model = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 1000)
            ).to(device).eval().requires_grad_(False)
            return model, ToTensor(), "test-weights"

        with patch.object(main.AutoencoderKL, "from_pretrained", return_value=TinyVAE()), patch.object(
            main, "load_classifier", side_effect=load_classifier
        ):
            main.main([
                "--device", "cpu", "--data-dir", str(self.images),
                "--output-dir", str(self.output), *extra,
            ])

    def test_progress_and_experiment_artifacts(self):
        console = io.StringIO()
        with patch("sys.stderr", console):
            self.run_main("--transfer-model", "resnet152", "--limit", "1")
        runs = list((self.output / "resnet50").iterdir())
        self.assertEqual(len(runs), 1)
        run = runs[0]
        text = (run / "run.log").read_text()
        for message in (
            "Starting experiment", "class_id=2", "VAE ready",
            "dtype=torch.float32", "Step 1/3 (pre-update)",
            "Step 2/3 (pre-update)", "Step 3/3 (pre-update)",
            "eot_ce=", "penalty=", "pixel_linf=",
            "Applied latent smoothing after step 2", "Saved adversarial image",
            "Target results", "Experiment completed",
        ):
            self.assertIn(message, text)
            self.assertIn(message, console.getvalue())
        self.assertEqual(len(list((run / "images").glob("*.png"))), 1)
        manifest = json.loads((run / "run.json").read_text())
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["settings"]["steps"], 3)
        results = json.loads((run / "results.json").read_text())
        self.assertEqual(results["resnet152"]["total"], 1)
        self.assertEqual(
            len((run / "resnet152_predictions.csv").read_text().splitlines()), 2
        )

    def test_all_targets_and_all_images(self):
        self.run_main("--surrogate", "vgg16", "--transfer-model", "all")
        run = next((self.output / "vgg16").iterdir())
        self.assertEqual(len(list((run / "images").glob("*.png"))), 2)
        results = json.loads((run / "results.json").read_text())
        self.assertEqual(set(results), set(main.TARGETS))
        self.assertTrue(all(result["total"] == 2 for result in results.values()))

    def test_nonfinite_loss_logged_without_saving_or_evaluation(self):
        with patch("generate_adversarial_image.eot_loss", return_value=torch.tensor(float("nan"))):
            with self.assertRaisesRegex(FloatingPointError, "step 1/3"):
                self.run_main("--transfer-model", "resnet50")
        run = next((self.output / "resnet50").iterdir())
        text = (run / "run.log").read_text()
        self.assertIn("Non-finite loss at step 1/3", text)
        self.assertIn("Experiment failed", text)
        self.assertNotIn("Saved adversarial image", text)
        self.assertEqual(list((run / "images").glob("*.png")), [])
        self.assertFalse((run / "results.json").exists())
        self.assertEqual(json.loads((run / "run.json").read_text())["status"], "failed")

    def test_invalid_cli_arguments(self):
        for args in (
            ["--transfer-model", "unknown"],
            ["--transfer-model", "all", "--limit", "0"],
        ):
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                main.parse_args(args)


class MetricsTests(unittest.TestCase):
    def test_asr_excludes_existing_mistakes(self):
        metrics = main.calculate_metrics(100, 80, 20, 60)
        self.assertEqual(metrics["asr_pct"], 80)
        self.assertEqual(metrics["successful_attacks"], 80)
        self.assertEqual(metrics["clean_correct_asr_pct"], 75)
        self.assertEqual(metrics["adversarial_misclassification_rate_pct"], 80)

    def test_asr_undefined_when_no_clean_images_are_correct(self):
        metrics = main.calculate_metrics(2, 0, 0, 0)
        self.assertIsNone(metrics["clean_correct_asr_pct"])
        self.assertEqual(metrics["adversarial_misclassification_rate_pct"], 100)

    def test_target_predictions_and_saved_png_preprocessing(self):
        class Predictions(nn.Module):
            def forward(self, batch):
                return torch.tensor(
                    [[0., 0., 4.], [4., 0., 0.]]
                    if batch[0].mean() < 0.5
                    else [[4., 0., 0.], [4., 0., 0.]]
                )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            image_dir = path / "adversarial"
            image_dir.mkdir()
            samples = []
            for index, value in enumerate((0, 255)):
                original = path / f"{index}_label2.png"
                Image.new("RGB", (32, 32), (value,) * 3).save(original)
                Image.new("RGB", (32, 32)).save(image_dir / f"{original.stem}_adversarial.png")
                samples.append((original, 2))
            result = main.evaluate_target(
                Predictions(), Compose([Resize((32, 32)), ToTensor()]),
                samples, image_dir, path / "predictions.csv", "cpu"
            )
            self.assertEqual(result["clean_correct"], 1)
            self.assertEqual(result["adversarial_correct"], 0)
            self.assertEqual(result["successful_attacks_on_clean_correct"], 1)
            self.assertEqual(result["clean_correct_asr_pct"], 100)
            self.assertEqual(result["adversarial_misclassification_rate_pct"], 100)


class EotTransformTests(unittest.TestCase):
    def test_every_sample_composes_resize_interpolation_and_crop(self):
        image = torch.rand(1, 3, 256, 256, requires_grad=True)
        torch.manual_seed(0)
        views = [apply_perturbation(image, 224) for _ in range(200)]

        # Each sampled transform ends in a crop, so the classifier input size is fixed.
        for view in views:
            self.assertEqual(view.shape[-2:], (224, 224))
        # Resize, kernel and jitter are all resampled, so views must differ.
        signatures = {tuple(v.detach().flatten()[:8].tolist()) for v in views}
        self.assertGreater(len(signatures), 150)

        views[0].sum().backward()
        self.assertIsNotNone(image.grad)
        self.assertGreater(image.grad.abs().sum().item(), 0)

    def test_accepts_unbatched_and_degenerate_parameters(self):
        self.assertEqual(
            apply_perturbation(torch.rand(3, 256, 256), 224).shape, (3, 224, 224)
        )
        # Upscaling happens before the crop, so inputs below the crop size work.
        self.assertEqual(
            apply_perturbation(torch.rand(1, 3, 64, 64), 224).shape, (1, 3, 224, 224)
        )
        self.assertEqual(
            apply_perturbation(torch.rand(1, 3, 256, 256), 224, max_scale=0, jitter=0).shape,
            (1, 3, 224, 224),
        )
        with self.assertRaises(ValueError):
            apply_perturbation(torch.rand(256, 256), 224)

    def test_penalty_gradient_survives_out_of_range_pixels(self):
        decoded = torch.tensor([-1.4, 0.0, 1.4], requires_grad=True)
        adversarial = decoded / 2 + 0.5

        # Algorithm 1 keeps the raw decoder output, so violations stay visible.
        self.assertTrue(
            torch.allclose(adversarial, torch.tensor([-0.2, 0.5, 1.2]), atol=1e-6)
        )
        soft_penalty(adversarial, torch.full((3,), 0.5), 16 / 255).backward()
        # A clamp here would zero both out-of-range gradients.
        assert decoded.grad is not None
        self.assertLess(decoded.grad[0].item(), 0)
        self.assertGreater(decoded.grad[2].item(), 0)


class SingleImageTests(unittest.TestCase):
    def test_function_uses_passed_models_and_settings_without_saving(self):
        image = Image.new("RGB", (32, 32), (100, 150, 200))
        vae = TinyVAE()
        surrogate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 1000)
        )
        with patch.object(vae, "decode", wraps=vae.decode) as decode:
            result = generate_adversarial_image(
                vae, surrogate, image, 2,
                steps=2, k=1, epsilon=8 / 255, penalty_weight=10,
                learning_rate=0.1, smoothing_interval=2, kernel_size=3,
                sigma=0.5, width=32, height=32, crop_size=16,
                mean=cfg.IMAGENET_MEAN, std=cfg.IMAGENET_STD, log_interval=1,
            )
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, (32, 32))
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(decode.call_count, 3)
        self.assertFalse(vae.training)
        self.assertFalse(surrogate.training)
        self.assertTrue(all(parameter.grad is None for parameter in surrogate.parameters()))


if __name__ == "__main__":
    unittest.main()
