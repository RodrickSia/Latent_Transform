import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import analyze_all_runs


class BatchAnalysisTests(unittest.TestCase):
    def test_only_completed_unreported_runs_are_analyzed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, status, results, report in (
                ("ready", "completed", True, False),
                ("reported", "completed", True, True),
                ("running", "running", True, False),
                ("failed", "failed", True, False),
                ("missing-results", "completed", False, False),
            ):
                run = root / "resnet50" / name
                run.mkdir(parents=True)
                (run / "run.json").write_text(json.dumps({"status": status}))
                if results:
                    (run / "results.json").write_text("{}")
                if report:
                    (run / "report.pdf").write_bytes(b"existing report")
            with patch.object(
                analyze_all_runs.subprocess, "run", return_value=SimpleNamespace(returncode=0)
            ) as process, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(analyze_all_runs.main(["--output-dir", directory]), 0)
            process.assert_called_once()
            self.assertEqual(Path(process.call_args.args[0][-1]), root / "resnet50" / "ready")
            self.assertEqual((root / "resnet50" / "reported" / "report.pdf").read_bytes(), b"existing report")

    def test_analysis_failures_do_not_stop_remaining_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("a", "b"):
                run = root / name
                run.mkdir()
                (run / "run.json").write_text('{"status": "completed"}')
                (run / "results.json").write_text("{}")
            with patch.object(
                analyze_all_runs.subprocess, "run",
                side_effect=[SimpleNamespace(returncode=1), SimpleNamespace(returncode=0)],
            ) as process, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(analyze_all_runs.main(["--output-dir", directory]), 1)
            self.assertEqual(process.call_count, 2)

    def test_invalid_manifest_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "run.json").write_text("invalid JSON")
            with patch.object(analyze_all_runs.subprocess, "run") as process, contextlib.redirect_stderr(
                io.StringIO()
            ) as errors:
                self.assertEqual(analyze_all_runs.main(["--output-dir", directory]), 1)
            process.assert_not_called()
            self.assertIn("ERROR reading", errors.getvalue())
