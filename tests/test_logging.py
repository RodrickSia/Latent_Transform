import io
import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image
from torch import nn

import config as cfg
import main


class TinyVAE(nn.Module):
    dtype = torch.float32

    def encode(self, image, return_dict=False):
        latent = nn.functional.avg_pool2d(image.float(), 8)
        return (SimpleNamespace(sample=lambda: latent),)

    def decode(self, latent, return_dict=False):
        return (nn.functional.interpolate(latent, size=(32, 32)).tanh(),)


class LoggingTests(unittest.TestCase):
    def setUp(self):
        self.root = logging.getLogger()
        self.previous_handlers = self.root.handlers[:]
        self.previous_level = self.root.level
        self.root.handlers = []
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.images = self.path / "images"
        self.images.mkdir()
        Image.new("RGB", (32, 32), (100, 150, 200)).save(
            self.images / "00000_label2.png"
        )
        self.output = self.path / "output"
        self.config_patch = patch.multiple(
            cfg,
            DEVICE="cpu",
            IMAGENET_DIR=self.images,
            IMAGE_INDEX=0,
            OUTPUT_DIR=self.output,
            LOG_INTERVAL=2,
            T=3,
            N=2,
            K=1,
            H=32,
            W=32,
            CROP_SIZE=16,
        )
        self.config_patch.start()
        self.addCleanup(self.config_patch.stop)

    def tearDown(self):
        for handler in self.root.handlers:
            handler.close()
        self.root.handlers = self.previous_handlers
        self.root.setLevel(self.previous_level)

    def run_main(self):
        surrogate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 1000)
        )
        with patch.object(main.AutoencoderKL, "from_pretrained", return_value=TinyVAE()), patch.object(
            main, "resnet50", return_value=surrogate
        ):
            main.main()

    def test_progress_written_to_console_and_file(self):
        console = io.StringIO()
        with patch("sys.stderr", console):
            self.run_main()
        text = (self.output / "attack.log").read_text()
        for message in (
            "Starting attack",
            "class_id=2",
            "VAE ready",
            "Surrogate ready",
            "dtype=torch.float32",
            "Step 1/3 (pre-update)",
            "Step 2/3 (pre-update)",
            "Step 3/3 (pre-update)",
            "eot_ce=",
            "penalty=",
            "pixel_linf=",
            "Applied latent smoothing after step 2",
            "Saved adversarial image",
        ):
            self.assertIn(message, text)
            self.assertIn(message, console.getvalue())
        self.assertTrue((self.output / "00000_label2_adversarial.png").is_file())

    def test_nonfinite_loss_logged_and_raised_without_saving(self):
        with patch.object(main, "eot_loss", return_value=torch.tensor(float("nan"))):
            with self.assertRaisesRegex(FloatingPointError, "step 1/3"):
                self.run_main()
        text = (self.output / "attack.log").read_text()
        self.assertIn("ERROR", text)
        self.assertIn("Non-finite loss at step 1/3", text)
        self.assertNotIn("Saved adversarial image", text)
        self.assertEqual(list(self.output.glob("*.png")), [])


if __name__ == "__main__":
    unittest.main()
