"""Compare several runs side by side in one PDF.

Pages:
    1. Settings and perturbation-quality table, plus quality bar charts.
    2. Transfer ASR per target, grouped by run.
    3. Mean loss progression of every run overlaid on shared axes.
    4. The same source images rendered once per run, for visual comparison.

Example:
    PYTHONPATH=src python src/compare_runs.py \
        --run-dir outputs/resnet50/<a> --run-dir outputs/resnet50/<b>
"""

import argparse
import json
from pathlib import Path

import numpy as np

import config as cfg
from analyze_run import (
    PAPER_ASR,
    PAPER_PSNR_DB,
    aggregate_traces,
    class_names,
    load_pair,
    panel_spec,
    parse_log,
    quality_metrics,
    read_predictions,
)


QUALITY_BARS = [
    ("psnr_db", "PSNR (dB, higher = less visible)"),
    ("l1_mean_abs_error_255", "L1 mean abs error (/255)"),
    ("l2_rmse_255", "L2 RMSE (/255)"),
    ("linf_255", "Linf (/255)"),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, action="append", required=True, dest="run_dirs",
        help="Repeat once per run. At least two are needed for a comparison.",
    )
    parser.add_argument("--label", action="append", dest="labels", help="One per run.")
    parser.add_argument("--samples", type=int, default=5, help="Shared images to render.")
    parser.add_argument(
        "--cases", type=int, default=10, help="Samples per agreement category.",
    )
    parser.add_argument(
        "--case-target",
        help="Target whose predictions define success. Default: a transfer target "
        "evaluated by every run, otherwise the surrogate.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/comparison.pdf"),
        help="Destination PDF.",
    )
    args = parser.parse_args(argv)
    if len(args.run_dirs) < 2:
        parser.error("Pass --run-dir at least twice.")
    if args.labels and len(args.labels) != len(args.run_dirs):
        parser.error("--label must be given once per --run-dir.")
    if args.samples <= 0:
        parser.error("--samples must be positive")
    if args.cases <= 0:
        parser.error("--cases must be positive")
    return args


def auto_labels(runs: list[dict]) -> list[str]:
    """Name each run by whatever actually differs between the runs compared.

    Spelling out every setting makes legends unreadable, so shared values are
    dropped: comparing one CE run against one margin run yields "ce" and
    "margin k0" rather than repeating the surrogate, lambda and image count on
    both. Timestamps are appended only if the short names would collide.
    """
    def describe(run: dict) -> dict[str, str]:
        settings = run["settings"]
        objective = settings.get("objective", "ce")
        kappa = settings.get("kappa")
        return {
            "surrogate": run["manifest"]["surrogate"],
            "objective": objective,
            "kappa": f"k{kappa:g}" if objective == "margin" and kappa is not None else "",
            "penalty": settings.get("penalty_norm", "l1"),
            "lambda": f"lam{settings['penalty_weight']:g}",
            "lr": f"lr{settings['learning_rate']:g}",
            "steps": f"T{settings['steps']}",
            "images": f"n{len(run['manifest'].get('images', []))}",
        }

    described = [describe(run) for run in runs]
    varying = [key for key in described[0] if len({d[key] for d in described}) > 1]
    labels = [" ".join(filter(None, (d[key] for key in varying))) for d in described]
    if len(set(labels)) < len(labels):
        labels = [
            f"{label} {run['run_dir'].name[:13]}".strip()
            for label, run in zip(labels, runs)
        ]
    return labels


def collect(run_dir: Path) -> dict:
    """Read every artifact of one run into a single record."""
    manifest = json.loads((run_dir / "run.json").read_text())
    settings = manifest["settings"]
    size = (settings["width"], settings["height"])
    dataset = Path(manifest["dataset"])
    results_path = run_dir / "results.json"
    results = json.loads(results_path.read_text()) if results_path.is_file() else {}
    traces, objective = parse_log(run_dir / "run.log")
    return {
        "run_dir": run_dir,
        "manifest": manifest,
        "settings": settings,
        "size": size,
        "dataset": dataset,
        "results": results.get("results", results),
        "objective": objective,
        "traces": traces,
        "summary": aggregate_traces(traces),
        "metrics": quality_metrics(run_dir, dataset, size),
        "images": sorted(
            f"{path.stem.removesuffix('_adversarial')}.png"
            for path in (run_dir / "images").glob("*_adversarial.png")
        ),
    }


def pick_case_target(runs: list[dict]) -> str:
    """A target every run evaluated, preferring a transfer model."""
    evaluated = [set(run["results"]) for run in runs]
    if not evaluated or not set.intersection(*evaluated):
        raise ValueError("The runs share no evaluated target; pass --case-target.")
    shared = set.intersection(*evaluated)
    surrogates = {run["manifest"]["surrogate"] for run in runs}
    # Keep the evaluation order of the first run rather than sorting alphabetically.
    ordered = [name for name in runs[0]["results"] if name in shared]
    transfer = [name for name in ordered if name not in surrogates]
    return (transfer or ordered)[0]


def case_groups(
    runs: list[dict], target: str
) -> dict[tuple[bool, ...], list[tuple[str, list[dict[str, str]]]]]:
    """Group shared images by which runs flipped ``target`` and which did not.

    Only images that ``target`` already classified correctly when clean are
    kept: an image the target was wrong about to begin with cannot be flipped,
    so counting it would reward or punish a run for nothing it did.
    """
    lookups = [
        {row["image"]: row for row in read_predictions(run["run_dir"], target)}
        for run in runs
    ]
    groups: dict[tuple[bool, ...], list[tuple[str, list[dict[str, str]]]]] = {}
    for name in sorted(set.intersection(*(set(lookup) for lookup in lookups))):
        rows = [lookup[name] for lookup in lookups]
        if any(row["clean_prediction"] != row["label"] for row in rows):
            continue
        key = tuple(row["adversarial_prediction"] != row["label"] for row in rows)
        groups.setdefault(key, []).append((name, rows))
    return groups


def describe_key(key: tuple[bool, ...], labels: list[str]) -> str:
    if all(key):
        return "Both succeed" if len(key) == 2 else "All runs succeed"
    if not any(key):
        return "Both fail" if len(key) == 2 else "All runs fail"
    short = list(labels)
    won = " + ".join(name for name, flipped in zip(short, key) if flipped)
    lost = " + ".join(name for name, flipped in zip(short, key) if not flipped)
    return f"{won} succeeds  /  {lost} fails"


def draw_case_grid(
    figure,
    entries: list[tuple[str, list[dict[str, str]]]],
    runs: list[dict],
    labels: list[str],
    categories: list[str],
    target: str,
    title: str,
    total: int,
) -> None:
    """One row per image: the clean source, then each run's adversarial result."""
    axes = figure.subplots(len(entries), len(runs) + 1, squeeze=False)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    for row, (name, predictions) in enumerate(entries):
        clean, _ = load_pair(
            runs[0]["run_dir"], runs[0]["dataset"], name, runs[0]["size"]
        )
        axes[row][0].imshow(np.clip(clean, 0, 1))
        truth = categories[int(predictions[0]["label"])][:22]
        axes[row][0].set_title(f"clean -> {truth}", fontsize=7)
        axes[row][0].set_ylabel(f"{Path(name).stem[:20]}\ntrue: {truth}", fontsize=6)
        for column, (run, prediction) in enumerate(zip(runs, predictions), 1):
            _, adversarial = load_pair(
                run["run_dir"], run["dataset"], name, run["size"]
            )
            axes[row][column].imshow(np.clip(adversarial, 0, 1))
            flipped = prediction["adversarial_prediction"] != prediction["label"]
            predicted = categories[int(prediction["adversarial_prediction"])][:22]
            axes[row][column].set_title(
                f"{'FLIPPED' if flipped else 'resisted'} -> {predicted}",
                fontsize=7, color="C3" if flipped else "0.35",
            )
            if row == 0:
                axes[row][column].set_xlabel(labels[column - 1], fontsize=6)
                axes[row][column].xaxis.set_label_position("top")
    figure.suptitle(
        f"{title}   on {target}   ({len(entries)} of {total} such images)", fontsize=12
    )


def draw_quality(figure, runs: list[dict], labels: list[str]) -> None:
    grid = figure.add_gridspec(2, 1, height_ratios=(1, 1.1), hspace=0.35)
    text_axis = figure.add_subplot(grid[0])
    text_axis.axis("off")
    header = f"{'run':<24}{'obj':>7}{'kappa':>7}{'pen':>5}{'lam':>6}{'n':>6}"
    header += f"{'PSNR':>8}{'L1':>7}{'L2':>7}{'Linf':>7}   directory"
    lines = [header, "-" * len(header)]
    for run, label in zip(runs, labels):
        settings, metrics = run["settings"], run["metrics"]
        lines.append(
            f"{label[:23]:<24}"
            f"{settings.get('objective', 'ce'):>7}"
            f"{settings.get('kappa') or 0.0:>7.1f}"
            f"{settings.get('penalty_norm', 'l1'):>5}"
            f"{settings['penalty_weight']:>6.1f}"
            f"{metrics['images']:>6d}"
            f"{metrics['psnr_db']:>8.2f}"
            f"{metrics['l1_mean_abs_error_255']:>7.2f}"
            f"{metrics['l2_rmse_255']:>7.2f}"
            f"{metrics['linf_255']:>7.1f}"
            f"   {run['run_dir'].name}"
        )
    lines += ["", f"paper reference PSNR {PAPER_PSNR_DB} dB;  "
                  f"epsilon {cfg.EPSILON * 255:.0f}/255 (L1/L2 in /255 units)"]
    text_axis.text(0, 1, "\n".join(lines), va="top", fontsize=8.5, family="monospace")
    text_axis.set_title("Perturbation quality", loc="left", fontsize=13)

    bar_grid = grid[1].subgridspec(1, len(QUALITY_BARS), wspace=0.45)
    positions = np.arange(len(runs))
    for column, (key, title) in enumerate(QUALITY_BARS):
        axis = figure.add_subplot(bar_grid[column])
        values = [run["metrics"][key] for run in runs]
        axis.bar(positions, values, color=[f"C{index}" for index in positions])
        if key == "psnr_db":
            axis.axhline(PAPER_PSNR_DB, color="k", linestyle="--", linewidth=1)
        for position, value in zip(positions, values):
            axis.annotate(
                f"{value:.2f}", (position, value), ha="center", va="bottom", fontsize=7
            )
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, fontsize=7, rotation=15, ha="right")
        axis.set_title(title, fontsize=8)
        axis.grid(axis="y", alpha=0.3)


def draw_asr(figure, runs: list[dict], labels: list[str]) -> None:
    targets: list[str] = []
    for run in runs:
        for name in run["results"]:
            if name not in targets:
                targets.append(name)
    axis = figure.subplots()
    if not targets:
        axis.axis("off")
        axis.set_title("No results.json in any run", loc="left")
        return

    surrogates = {run["manifest"]["surrogate"] for run in runs}
    paper = PAPER_ASR.get(next(iter(surrogates))) if len(surrogates) == 1 else None
    series: list[tuple[str, dict]] = [
        (label, run["results"]) for label, run in zip(labels, runs)
    ]
    if paper:
        series.append(("paper", {name: {"asr_pct": paper[name]} for name in targets
                                 if name in paper}))
    positions = np.arange(len(targets))
    width = 0.8 / len(series)
    for index, (label, results) in enumerate(series):
        offset = (index - (len(series) - 1) / 2) * width
        values = [results.get(name, {}).get("asr_pct", np.nan) for name in targets]
        axis.bar(
            positions + offset, values, width,
            label=label.replace("\n", " "),
            alpha=0.6 if label == "paper" else 0.95,
            color="0.3" if label == "paper" else f"C{index}",
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(targets, rotation=30, ha="right", fontsize=8)
    axis.set_ylabel("ASR %")
    axis.set_ylim(0, 108)
    axis.grid(axis="y", alpha=0.3)
    axis.legend(fontsize=7, ncol=min(len(series), 4))
    means = ", ".join(
        f"{label} {np.nanmean([run['results'].get(n, {}).get('asr_pct', np.nan) for n in targets]):.1f}"
        for label, run in zip(labels, runs)
    )
    axis.set_title(f"Transfer ASR by target   mean: {means}", loc="left", fontsize=11)


def draw_mean_comparison(figure, runs: list[dict], labels: list[str]) -> None:
    axes = figure.subplots(2, 2)
    # Objectives may differ between runs, so resolve each panel per run.
    titles = [title for _, title in panel_spec("objective")]
    keys = ["loss", None, "penalty", "pixel_linf"]
    for axis, key, title in zip(axes.flat, keys, titles):
        for index, (run, label) in enumerate(zip(runs, labels)):
            metric = key or run["objective"]
            stats = run["summary"].get(metric)
            if stats is None:
                continue
            axis.plot(
                stats["step"], stats["mean"], linewidth=1.8, color=f"C{index}",
                label=f"{label} ({metric})",
            )
            axis.fill_between(
                stats["step"], stats["mean"] - stats["std"],
                stats["mean"] + stats["std"], alpha=0.15, color=f"C{index}",
            )
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("optimization step")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=6)
    axes.flat[3].axhline(cfg.EPSILON, color="k", linestyle="--", linewidth=1)
    figure.suptitle("Mean loss progression over all images, per run")


def draw_samples(figure, runs: list[dict], labels: list[str], names: list[str]) -> None:
    axes = figure.subplots(len(runs) + 1, len(names), squeeze=False)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    for column, name in enumerate(names):
        clean, _ = load_pair(runs[0]["run_dir"], runs[0]["dataset"], name, runs[0]["size"])
        axes[0][column].imshow(np.clip(clean, 0, 1))
        axes[0][column].set_title(Path(name).stem[:18], fontsize=7)
        for row, run in enumerate(runs, 1):
            _, adversarial = load_pair(
                run["run_dir"], run["dataset"], name, run["size"]
            )
            axes[row][column].imshow(np.clip(adversarial, 0, 1))
            if column == 0:
                axes[row][column].set_ylabel(labels[row - 1], fontsize=6)
    axes[0][0].set_ylabel("clean", fontsize=8)
    figure.suptitle("Same sources, one row per run")


def build(
    args: argparse.Namespace,
) -> tuple[list[str], list[dict], list[str], str, dict[str, int]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    runs = [collect(run_dir) for run_dir in args.run_dirs]
    labels = args.labels or auto_labels(runs)
    shared = sorted(set.intersection(*(set(run["images"]) for run in runs)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output) as pdf:
        quality = plt.figure(figsize=(13, 8.5))
        draw_quality(quality, runs, labels)
        pdf.savefig(quality)
        plt.close(quality)

        asr = plt.figure(figsize=(13, 7))
        draw_asr(asr, runs, labels)
        asr.tight_layout()
        pdf.savefig(asr)
        plt.close(asr)

        means = plt.figure(figsize=(13, 8))
        draw_mean_comparison(means, runs, labels)
        means.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(means)
        plt.close(means)

        if shared:
            selected = shared[: args.samples]
            figure = plt.figure(
                figsize=(2.2 * len(selected) + 1.5, 2.3 * (len(runs) + 1) + 0.8)
            )
            draw_samples(figure, runs, labels, selected)
            figure.tight_layout(rect=(0, 0, 1, 0.95))
            pdf.savefig(figure)
            plt.close(figure)

        target = args.case_target or pick_case_target(runs)
        groups = case_groups(runs, target)
        categories = class_names()
        # All-succeed and all-fail first, then the disagreements.
        order = sorted(groups, key=lambda key: (0 if len(set(key)) == 1 else 1, key))
        counts = {}
        for key in order:
            entries = groups[key][: args.cases]
            title = describe_key(key, labels)
            counts[title] = len(groups[key])
            figure = plt.figure(
                figsize=(2.9 * (len(runs) + 1) + 1, 2.5 * len(entries) + 1.1)
            )
            draw_case_grid(
                figure, entries, runs, labels, categories, target, title,
                len(groups[key]),
            )
            figure.tight_layout(rect=(0, 0, 1, 0.95))
            pdf.savefig(figure)
            plt.close(figure)

    return shared, runs, labels, target, counts


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    shared, runs, labels, target, counts = build(args)
    for label, run in zip(labels, runs):
        metrics = run["metrics"]
        asr = [
            values["asr_pct"] for values in run["results"].values() if "asr_pct" in values
        ]
        summary = f"mean ASR {np.mean(asr):.1f}" if asr else "no results.json"
        print(
            f"{label:<24} n={metrics['images']:<5} "
            f"PSNR {metrics['psnr_db']:6.2f} dB  "
            f"L1 {metrics['l1_mean_abs_error_255']:5.2f}  "
            f"L2 {metrics['l2_rmse_255']:5.2f}  "
            f"Linf {metrics['linf_255']:6.2f}  {summary}"
        )
    if not shared:
        print("No source image is common to every run; sample page skipped.")
    print(f"\nAgreement on {target} (clean-correct images only):")
    for title, count in sorted(counts.items(), key=lambda item: -item[1]):
        print(f"  {title:<48} {count}")
    print(f"Comparison: {args.output}")


if __name__ == "__main__":
    main()
