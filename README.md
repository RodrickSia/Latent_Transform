# LTA experiment

Generate images using a surrogate and evaluate one transfer model:

```bash
uv run python src/main.py --surrogate resnet50 --transfer-model vit_b_16 --limit 10
```

Omit `--limit` to process all downloaded images. To evaluate every paper target:

```bash
uv run python src/main.py --surrogate vgg16 --transfer-model all
```

Surrogates: `resnet50`, `resnet152`, `vgg16`.
Targets: `resnet50`, `resnet152`, `vgg16`, `mobilenet_v2`, `inception_v3`,
`vit_b_16`, `pit_b`, `visformer_small`, `swin_t`, or `all`.
The transfer-model argument selects the evaluation classifier, not a desired
class: this remains an untargeted attack. Models use torchvision default
pretrained weights, except PiT-B and Visformer-S, which use timm.
These weight versions are implementation choices, not verified author checkpoints.

Each run saves to `outputs/<surrogate>/<timestamp>/`:

- `images/`: generated adversarial PNGs.
- `run.log`: progress, loss components, errors, and target metrics.
- `run.json`: settings, seed, image list, weights, metric definitions, and run status.
- `results.json`: per-target counts, accuracy, and Attack Success Rate (ASR).
- `<target>_predictions.csv`: labels and clean/adversarial predictions for each image.

Primary `asr_pct` is `100 * adversarial misclassifications / total attack attempts`.
The separate `clean_correct_asr_pct` counts only images the target classified
correctly before attack; it is `null` when no clean images are correct.
Evaluation uses saved PNGs and the same weight-specific preprocessing for clean
and adversarial images. Clean images first receive the attack's 256x256 resize.

Attack hyperparameters remain in `src/config.py`. Epsilon is a soft penalty
threshold, not a guaranteed maximum pixel difference. Existing run folders are
not overwritten. Pretrained weights are downloaded on first use; `all` loads
evaluation models sequentially after generation to reduce GPU memory use.

For single-image use, import `generate_adversarial_image` from
`src/generate_adversarial_image.py`. It takes the loaded VAE, surrogate, PIL image,
integer label, and explicit attack settings, and returns a PIL image. The caller
sets the random seed and saves the returned image.

Tests (small test models, no pretrained downloads):

```bash
PYTHONPATH=src uv run python -m unittest discover -s tests -v
```