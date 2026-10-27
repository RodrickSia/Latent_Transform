from pathlib import Path


DEVICE = "cuda"
VAE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
IMAGENET_DIR = Path(__file__).resolve().parents[1] / "data" / "imagenet_eval"
IMAGE_INDEX = 0  # Select one image from the sorted dataset filenames.
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

T = 150  # adversarial optimization steps
K = 4  # sampled perturbations per EOT loss
N = 10  # smoothing interval (steps)
EPSILON = 16 / 255
LAMBDA = 10.0  # soft penalty weight
LR = 0.1
SEED = 42

H = 256
W = 256
CROP_SIZE = 224

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SMOOTH_KERNEL_SIZE = 3  # must be odd
SMOOTH_SIGMA = 0.5
