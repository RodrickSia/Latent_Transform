import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate PDFs for completed, unreported runs.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    if not output_dir.is_dir():
        parser.error(f"Output directory does not exist: {output_dir}")

    failed = False
    for manifest_path in sorted(output_dir.rglob("run.json")):
        run_dir = manifest_path.parent
        if (run_dir / "report.pdf").exists():
            print(f"SKIP (report exists): {run_dir}", flush=True)
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError) as error:
            print(f"ERROR reading {manifest_path}: {error}", file=sys.stderr, flush=True)
            failed = True
            continue
        if not isinstance(manifest, dict):
            print(f"ERROR: {manifest_path} must contain a JSON object", file=sys.stderr, flush=True)
            failed = True
            continue
        if manifest.get("status") != "completed" or not (run_dir / "results.json").is_file():
            print(f"SKIP (run not finished): {run_dir}", flush=True)
            continue

        print(f"ANALYZE: {run_dir}", flush=True)
        result = subprocess.run(
            [sys.executable, str(ROOT / "src" / "analyze_run.py"), "--run-dir", str(run_dir)],
            cwd=ROOT,
        )
        if result.returncode != 0:
            print(f"ERROR analyzing {run_dir}: exit {result.returncode}", file=sys.stderr, flush=True)
            failed = True

    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
