"""Aggregate every artifact of a run into one PDF report.

Pages:
    1. Overview      - settings, perturbation quality, transfer ASR vs the paper.
    2. Loss traces   - per-image optimization curves parsed from ``run.log``.
    3. Success cases - clean / adversarial / amplified difference triplets.
    4. Failure cases - the same, for images the target still classifies correctly.

Example:
    PYTHONPATH=src python src/analyze_run.py --run-dir outputs/resnet50/<stamp>
"""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

import config as cfg


NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|nan|inf|-inf"
IMAGE_LINE = re.compile(r"Generating image \d+/\d+ \| input=(\S+) \| class_id=(\d+)")
STEP_LINE = re.compile(r"Step (\d+)/\d+ \(pre-update\) \| (.*?) \| elapsed=")
FIELD = re.compile(rf"(\w+)=({NUMBER})")

# Table 1 of the paper, RN50 surrogate. Other surrogates are not transcribed.
PAPER_ASR = {
    "resnet50": {
        "resnet50": 100.0, "resnet152": 96.3, "vgg16": 92.8, "mobilenet_v2": 91.4,
        "inception_v3": 82.9, "vit_b_16": 71.3, "pit_b": 91.1,
        "visformer_small": 92.1, "swin_t": 91.4,
    }
}
PAPER_PSNR_DB = 22.26


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="Defaults to the most recent run.")
    parser.add_argument("--images", type=int, default=6, help="Images per trace plot.")
    parser.add_argument("--gallery", type=int, default=6, help="Images in the sample gallery.")
    parser.add_argument("--cases", type=int, default=4, help="Success/failure examples.")
    parser.add_argument(
        "--case-target", help="Target whose predictions pick the cases. Default: a "
        "transfer target if one was evaluated, otherwise the surrogate.",
    )
    parser.add_argument("--output-dir", type=Path, default=cfg.OUTPUT_DIR)
    args = parser.parse_args(argv)
    if min(args.images, args.cases, args.gallery) <= 0:
        parser.error("--images, --gallery and --cases must be positive")
    return args


def latest_run(output_dir: Path) -> Path:
    runs = [path for path in output_dir.glob("*/*") if (path / "run.log").is_file()]
    if not runs:
        raise FileNotFoundError(f"No run with a run.log under {output_dir}.")
    return max(runs, key=lambda path: path.stat().st_mtime)


def class_names() -> list[str]:
    from torchvision.models import get_model_weights

    # Static metadata on the weights enum; no checkpoint download.
    return list(get_model_weights("resnet50").DEFAULT.meta["categories"])


def parse_log(log_path: Path) -> tuple[dict[str, dict[str, list[float]]], str]:
    """Return {image: {metric: values}} plus the objective field name."""
    traces: dict[str, dict[str, list[float]]] = {}
    current: dict[str, list[float]] | None = None
    objective = "loss"
    for line in log_path.read_text(errors="replace").splitlines():
        started = IMAGE_LINE.search(line)
        if started:
            current = traces.setdefault(started.group(1), {})
            continue
        step = STEP_LINE.search(line)
        if not step or current is None:
            continue
        current.setdefault("step", []).append(float(step.group(1)))
        for name, value in FIELD.findall(step.group(2)):
            if name.startswith("eot_"):
                objective = name
            current.setdefault(name, []).append(float(value))
    return traces, objective


def aggregate_traces(
    traces: dict[str, dict[str, list[float]]]
) -> dict[str, dict[str, np.ndarray]]:
    """Pool every image's curve into a per-step mean, std and quartile band.

    Images are bucketed by step value rather than by position, so runs that
    stopped early still contribute to the steps they reached; ``count`` records
    how many images back each point.
    """
    pooled: dict[str, dict[float, list[float]]] = {}
    for trace in traces.values():
        steps = trace.get("step", [])
        for metric, values in trace.items():
            if metric == "step":
                continue
            buckets = pooled.setdefault(metric, {})
            for step, value in zip(steps, values):
                if np.isfinite(value):
                    buckets.setdefault(step, []).append(value)

    summary: dict[str, dict[str, np.ndarray]] = {}
    for metric, buckets in pooled.items():
        steps = sorted(buckets)
        columns = [np.asarray(buckets[step]) for step in steps]
        summary[metric] = {
            "step": np.asarray(steps),
            "mean": np.asarray([column.mean() for column in columns]),
            "std": np.asarray([column.std() for column in columns]),
            "p25": np.asarray([np.percentile(column, 25) for column in columns]),
            "p75": np.asarray([np.percentile(column, 75) for column in columns]),
            "count": np.asarray([column.size for column in columns]),
        }
    return summary


def panel_spec(objective: str) -> list[tuple[str, str]]:
    return [
        ("loss", "Total loss  (-objective + lambda * penalty)"),
        (objective, f"Surrogate objective  ({objective})"),
        ("penalty", "Soft pixel penalty"),
        ("pixel_linf", "Pixel Linf vs clean"),
    ]


def draw_mean_traces(figure, summary: dict, objective: str, total: int) -> None:
    axes = figure.subplots(2, 2)
    for axis, (metric, title) in zip(axes.flat, panel_spec(objective)):
        stats = summary.get(metric)
        if stats is None:
            axis.set_title(f"{title}  (not logged)", fontsize=10)
            continue
        step = stats["step"]
        axis.fill_between(
            step, stats["p25"], stats["p75"], alpha=0.2, label="interquartile"
        )
        axis.fill_between(
            step, stats["mean"] - stats["std"], stats["mean"] + stats["std"],
            alpha=0.2, label="mean +/- 1 std",
        )
        axis.plot(step, stats["mean"], linewidth=2, color="C3", label="mean")
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("optimization step")
        axis.grid(alpha=0.3)
        if stats["count"].min() < total:
            axis.annotate(
                f"min {stats['count'].min()} of {total} images at some steps",
                (0.02, 0.02), xycoords="axes fraction", fontsize=7, color="0.4",
            )
    axes.flat[3].axhline(cfg.EPSILON, color="k", linestyle="--", linewidth=1)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    figure.suptitle(f"Loss progression averaged over all {total} images")


def draw_gallery(
    figure, names: list[str], run_dir: Path, dataset: Path,
    size: tuple[int, int], rows_per_image: int = 3,
) -> None:
    """Clean / adversarial / amplified-difference columns, one per image."""
    axes = figure.subplots(rows_per_image, len(names), squeeze=False)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    for column, name in enumerate(names):
        clean, adversarial = load_pair(run_dir, dataset, name, size)
        difference = np.abs(adversarial - clean)
        peak = float(difference.max()) or 1.0
        panels = (
            (clean, "clean"),
            (adversarial, "adversarial"),
            (difference / peak, f"|diff| x{1 / peak:.1f}"),
        )
        for row, (image, label) in enumerate(panels):
            axes[row][column].imshow(np.clip(image, 0, 1))
            if column == 0:
                axes[row][column].set_ylabel(label, fontsize=9)
        axes[0][column].set_title(Path(name).stem[:18], fontsize=7)
    figure.suptitle(f"Adversarial samples side by side ({len(names)} images)")


def load_pair(run_dir: Path, dataset: Path, name: str, size: tuple[int, int]):
    """Return the clean and adversarial images as [0, 1] float arrays."""
    with Image.open(dataset / name) as handle:
        clean = np.asarray(handle.convert("RGB").resize(size), dtype=np.float64) / 255
    stem = Path(name).stem
    with Image.open(run_dir / "images" / f"{stem}_adversarial.png") as handle:
        adversarial = np.asarray(handle.convert("RGB"), dtype=np.float64) / 255
    return clean, adversarial


def quality_metrics(run_dir: Path, dataset: Path, size: tuple[int, int]) -> dict:
    """Average PSNR, L1, L2 and Linf between clean and adversarial images."""
    psnr, mae, rmse, l1_norm, l2_norm, linf = [], [], [], [], [], []
    for path in sorted((run_dir / "images").glob("*_adversarial.png")):
        name = f"{path.stem.removesuffix('_adversarial')}.png"
        if not (dataset / name).is_file():
            raise FileNotFoundError(f"Missing clean source {dataset / name}.")
        clean, adversarial = load_pair(run_dir, dataset, name, size)
        difference = adversarial - clean
        mse = float(np.mean(difference**2))
        # Both images are 8-bit, so a zero MSE means an identical file.
        psnr.append(float("inf") if mse == 0 else 10 * np.log10(1.0 / mse))
        mae.append(float(np.mean(np.abs(difference))))
        rmse.append(float(np.sqrt(mse)))
        l1_norm.append(float(np.sum(np.abs(difference))))
        l2_norm.append(float(np.sqrt(np.sum(difference**2))))
        linf.append(float(np.max(np.abs(difference))))

    if not psnr:
        raise FileNotFoundError(f"No adversarial PNGs under {run_dir / 'images'}.")
    finite = [value for value in psnr if np.isfinite(value)]
    return {
        "images": len(psnr),
        "psnr_db": float(np.mean(finite)) if finite else float("inf"),
        "l1_mean_abs_error": float(np.mean(mae)),
        "l1_mean_abs_error_255": float(np.mean(mae)) * 255,
        "l1_norm": float(np.mean(l1_norm)),
        "l2_rmse": float(np.mean(rmse)),
        "l2_rmse_255": float(np.mean(rmse)) * 255,
        "l2_norm": float(np.mean(l2_norm)),
        "linf": float(np.mean(linf)),
        "linf_255": float(np.mean(linf)) * 255,
        "epsilon_255": cfg.EPSILON * 255,
    }


def read_predictions(run_dir: Path, target: str) -> list[dict[str, str]]:
    path = run_dir / f"{target}_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"No predictions for {target} at {path}.")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def draw_overview(figure, manifest: dict, metrics: dict, results: dict) -> None:
    settings = manifest["settings"]
    surrogate = manifest["surrogate"]
    grid = figure.add_gridspec(2, 1, height_ratios=(1, 1.3), hspace=0.3)

    text_axis = figure.add_subplot(grid[0])
    text_axis.axis("off")
    psnr_note = f"   (paper {PAPER_PSNR_DB} dB)" if surrogate == "resnet50" else ""
    lines = [
        f"surrogate {surrogate}      objective {settings.get('objective', 'ce')}"
        f"      kappa {settings.get('kappa', 0.0)}      images {metrics['images']}",
        f"steps {settings['steps']}   K {settings['k']}   "
        f"lambda {settings['penalty_weight']}   lr {settings['learning_rate']}   "
        f"penalty {settings.get('penalty_norm', 'l1')}   "
        f"smoothing every {settings['smoothing_interval']} "
        f"({settings['kernel_size']}x{settings['kernel_size']}, sigma {settings['sigma']})",
        "",
        f"PSNR  {metrics['psnr_db']:6.2f} dB{psnr_note}",
        f"L1    {metrics['l1_mean_abs_error_255']:6.2f}/255 mean abs error"
        f"      norm {metrics['l1_norm']:.0f}",
        f"L2    {metrics['l2_rmse_255']:6.2f}/255 RMSE"
        f"      norm {metrics['l2_norm']:.2f}",
        f"Linf  {metrics['linf_255']:6.2f}/255      "
        f"epsilon {metrics['epsilon_255']:.0f}/255 (soft penalty, not a hard bound)",
    ]
    text_axis.text(
        0, 1, "\n".join(lines), va="top", ha="left", fontsize=11, family="monospace"
    )
    text_axis.set_title(
        f"Run overview  -  {manifest.get('status', '?')}", loc="left", fontsize=13
    )

    paper = PAPER_ASR.get(surrogate, {})
    targets = [name for name in results if not paper or name in paper]
    ours = [results[name]["asr_pct"] for name in targets]
    positions = np.arange(len(targets))
    asr_axis = figure.add_subplot(grid[1])
    width = 0.38 if paper else 0.6
    asr_axis.bar(positions - (width / 2 if paper else 0), ours, width, label="ours")
    if paper:
        asr_axis.bar(
            positions + width / 2, [paper[name] for name in targets], width,
            label="paper", alpha=0.65,
        )
    for position, value in zip(positions, ours):
        asr_axis.annotate(
            f"{value:.1f}", (position - (width / 2 if paper else 0), value),
            ha="center", va="bottom", fontsize=7,
        )
    asr_axis.set_xticks(positions)
    asr_axis.set_xticklabels(targets, rotation=30, ha="right", fontsize=8)
    asr_axis.set_ylabel("ASR %")
    asr_axis.set_ylim(0, 108)
    asr_axis.grid(axis="y", alpha=0.3)
    asr_axis.legend(fontsize=8)
    title = "Transfer ASR by target"
    if paper and targets:
        reference = np.mean([paper[name] for name in targets])
        title += (f"   mean {np.mean(ours):.1f} vs {reference:.1f}"
                  f"  ({np.mean(ours) - reference:+.1f})")
    asr_axis.set_title(title, loc="left", fontsize=13)


def draw_traces(figure, traces: dict, objective: str, limit: int) -> None:
    panels = panel_spec(objective)
    selected = list(traces.items())[:limit]
    axes = figure.subplots(2, 2)
    for axis, (metric, title) in zip(axes.flat, panels):
        for name, trace in selected:
            if metric in trace:
                axis.plot(trace["step"], trace[metric], linewidth=1.2, label=name)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("optimization step")
        axis.grid(alpha=0.3)
    axes.flat[3].axhline(cfg.EPSILON, color="k", linestyle="--", linewidth=1)
    axes.flat[3].annotate(
        f"epsilon = {cfg.EPSILON:.4f}", (0.02, cfg.EPSILON),
        xycoords=("axes fraction", "data"), fontsize=8, va="bottom",
    )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), fontsize=7)
    figure.suptitle(f"Optimization traces ({len(selected)} images)")


def draw_cases(
    figure, rows: list[dict[str, str]], run_dir: Path, dataset: Path,
    size: tuple[int, int], names: list[str], target: str, kind: str,
) -> None:
    axes = figure.subplots(len(rows), 3, squeeze=False)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    for index, row in enumerate(rows):
        clean, adversarial = load_pair(run_dir, dataset, row["image"], size)
        difference = np.abs(adversarial - clean)
        peak = float(difference.max()) or 1.0
        panels = (
            (clean, f"clean -> {names[int(row['clean_prediction'])][:24]}"),
            (adversarial, f"adversarial -> {names[int(row['adversarial_prediction'])][:24]}"),
            (difference / peak, f"|diff| normalized   Linf {peak * 255:.0f}/255"),
        )
        for column, (image, title) in enumerate(panels):
            axes[index][column].imshow(np.clip(image, 0, 1))
            axes[index][column].set_title(title, fontsize=8)
        axes[index][0].set_ylabel(
            f"{row['image']}\ntrue: {names[int(row['label'])][:20]}", fontsize=6
        )
    figure.suptitle(f"{kind} cases on {target}  ({len(rows)} shown)")


def build_report(args: argparse.Namespace, run_dir: Path) -> tuple[Path, dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    manifest = json.loads((run_dir / "run.json").read_text())
    settings = manifest["settings"]
    size = (settings["width"], settings["height"])
    dataset = Path(manifest["dataset"])
    results = json.loads((run_dir / "results.json").read_text())
    results = results.get("results", results)

    metrics = quality_metrics(run_dir, dataset, size)
    traces, objective = parse_log(run_dir / "run.log")

    target = args.case_target
    if target is None:
        transfer = [name for name in results if name != manifest["surrogate"]]
        target = transfer[0] if transfer else next(iter(results))
    predictions = read_predictions(run_dir, target)
    # Only images the target got right when clean are a fair test either way.
    successes = [row for row in predictions if row["success_on_clean_correct"] == "True"]
    failures = [
        row for row in predictions
        if row["clean_prediction"] == row["label"]
        and row["adversarial_prediction"] == row["label"]
    ]
    names = class_names()

    destination = run_dir / "report.pdf"
    with PdfPages(destination) as pdf:
        overview = plt.figure(figsize=(11, 8.5))
        draw_overview(overview, manifest, metrics, results)
        pdf.savefig(overview)
        plt.close(overview)

        trace_figure = plt.figure(figsize=(13, 8))
        draw_traces(trace_figure, traces, objective, args.images)
        trace_figure.tight_layout(rect=(0, 0.06, 1, 0.96))
        pdf.savefig(trace_figure)
        plt.close(trace_figure)

        mean_figure = plt.figure(figsize=(13, 8))
        draw_mean_traces(
            mean_figure, aggregate_traces(traces), objective, len(traces)
        )
        mean_figure.tight_layout(rect=(0, 0.06, 1, 0.96))
        pdf.savefig(mean_figure)
        plt.close(mean_figure)

        gallery = sorted(traces)[: args.gallery] or [
            f"{path.stem.removesuffix('_adversarial')}.png"
            for path in sorted((run_dir / "images").glob("*_adversarial.png"))
        ][: args.gallery]
        if gallery:
            figure = plt.figure(figsize=(2.2 * len(gallery) + 1, 7.2))
            draw_gallery(figure, gallery, run_dir, dataset, size)
            figure.tight_layout(rect=(0, 0, 1, 0.95))
            pdf.savefig(figure)
            plt.close(figure)

        for kind, rows in (("Success", successes), ("Failure", failures)):
            selected = rows[: args.cases]
            if not selected:
                continue
            figure = plt.figure(figsize=(9, 3.1 * len(selected) + 0.8))
            draw_cases(figure, selected, run_dir, dataset, size, names, target, kind)
            figure.tight_layout(rect=(0, 0, 1, 0.96))
            pdf.savefig(figure)
            plt.close(figure)

    metrics["case_target"] = target
    metrics["case_successes"] = len(successes)
    metrics["case_failures"] = len(failures)
    return destination, metrics


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = args.run_dir or latest_run(args.output_dir)
    report, metrics = build_report(args, run_dir)
    (run_dir / "quality.json").write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"Run     : {run_dir}")
    print(f"Images  : {metrics['images']}")
    print(f"PSNR    : {metrics['psnr_db']:.2f} dB")
    print(f"L1      : {metrics['l1_mean_abs_error_255']:.2f}/255 (norm {metrics['l1_norm']:.0f})")
    print(f"L2      : {metrics['l2_rmse_255']:.2f}/255 RMSE (norm {metrics['l2_norm']:.2f})")
    print(f"Linf    : {metrics['linf_255']:.2f}/255 (epsilon {metrics['epsilon_255']:.0f}/255)")
    print(f"Cases   : {metrics['case_successes']} flipped / {metrics['case_failures']} "
          f"resisted on {metrics['case_target']}")
    print(f"Report  : {report}")


if __name__ == "__main__":
    main()
