import copy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from unittest.mock import patch

import run_paired_experiment as paired
from isolate_revision import ROOT, write_csv, write_json


def graph(identity):
    return {
        "video_id": identity,
        "nodes": [
            {"id": identity, "type": "video", "field": "video", "text": ""},
            {"id": identity + "-text", "type": "aspect", "field": "hook", "text": "old"},
        ],
        "edges": [{"source": identity, "target": identity + "-text", "type": "video_has_aspect"}],
    }


def fixture():
    original = {
        "video_id": "video-a", "youtube_id": "abcdefghijk", "brand_family": "brand",
        "views": "100", "duration_seconds": "30", "age_days": "200",
        "captured_at": "2026-11-09T00:00:00+00:00", "hook": "Share happiness.",
        "group": "old-group", "split": "test",
    }
    revised = {
        **original, "hook": "", "original_hook": original["hook"],
        "group": "corrected-campaign", "split": "train",
    }
    return original, revised


class PairedInputTests(unittest.TestCase):
    def test_baseline_restores_original_text_but_keeps_corrected_group_and_split(self):
        original, revised = fixture()
        old_graph = graph("video-a")
        new_graph = copy.deepcopy(old_graph)
        new_graph["nodes"][1]["text"] = "new"
        rows, graphs = paired.assemble_inputs(
            [revised], {"video-a": new_graph}, [original], [old_graph],
            ["brand_family"], ["hook"],
        )
        self.assertEqual(rows["v2"][0]["hook"], original["hook"])
        self.assertEqual(rows["v3"][0]["hook"], "")
        for policy in paired.POLICIES:
            self.assertEqual(rows[policy][0]["group"], "corrected-campaign")
            self.assertEqual(rows[policy][0]["split"], "train")
        self.assertEqual(graphs["v2"][0]["nodes"][1]["text"], "old")
        self.assertEqual(graphs["v3"][0]["nodes"][1]["text"], "new")
        self.assertEqual(original["group"], "old-group")

    def test_quarantine_never_enters_either_policy(self):
        original, revised = fixture()
        pending = {**revised, "video_id": "pending", "split": "quarantine"}
        rows, _ = paired.assemble_inputs(
            [pending, revised], {"video-a": graph("video-a")}, [original], [graph("video-a")],
            ["brand_family"], ["hook"],
        )
        self.assertTrue(all(len(records) == 1 for records in rows.values()))

    def test_changed_control_and_text_provenance_are_rejected(self):
        original, revised = fixture()
        for field, value in (("views", "101"), ("age_days", "300"), ("original_hook", "invented")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                paired.assemble_inputs(
                    [{**revised, field: value}], {"video-a": graph("video-a")},
                    [original], [graph("video-a")], ["brand_family"], ["hook"],
                )

    def test_dangling_edges_rejected(self):
        invalid = graph("video-a")
        invalid["edges"][0]["target"] = "missing"
        with self.assertRaisesRegex(ValueError, "dangling"):
            paired.validate_graph(invalid)

    def test_staged_sources_keep_legacy_filter_only_in_v2(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            a = paired.stage_policy(base / "v2", "v2")
            b = paired.stage_policy(base / "v3", "v3")
            for name in ("pipeline.py", "revision.py"):
                self.assertEqual((a / "src" / name).read_bytes(), (ROOT / "src" / name).read_bytes())
                self.assertIn("from text_policy import response_free_text",
                              (b / "src" / name).read_text(encoding="utf-8"))
            self.assertFalse((a / "src" / "text_policy.py").exists())

    def test_prediction_alignment_is_by_identity_not_row_order(self):
        rows = [
            {"video_id": "a", "group": "g1", "views": "1"},
            {"video_id": "b", "group": "g2", "views": "2"},
        ]
        left, right = paired.align_predictions(rows, list(reversed(rows)), rows)
        self.assertEqual(left, right)
        for bad in ([rows[0]], [rows[0], rows[0]], [rows[0], {**rows[1], "views": "3"}]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                paired.align_predictions(rows, bad, rows)

    def test_input_hash_guard_detects_changes(self):
        original, revised = fixture()
        rows, graphs = paired.assemble_inputs(
            [revised], {"video-a": graph("video-a")}, [original], [graph("video-a")],
            ["brand_family"], ["hook"],
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            paired.prepare_run(output, rows, graphs, [], {}, smoke=True)
            paired.validate_run(output)
            (output / "v2" / "videos.csv").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "input/source changed"):
                paired.validate_run(output)

    def test_outputs_are_new_and_isolated(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(paired, "ROOT", Path(directory)):
            good = Path(directory) / "artifacts" / "paired_revision" / "new"
            self.assertEqual(paired.output_path(good), good.resolve())
            good.mkdir(parents=True)
            for path in (good, good.parent, Path(directory) / "artifacts" / "v2" / "new"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    paired.output_path(path)

    def test_run_requires_acknowledgement_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not-created"
            result = subprocess.run([
                sys.executable, str(ROOT / "Scripts" / "run_paired_experiment.py"),
                "--input", "does-not-exist", "--output", str(output), "--run",
            ], cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("acknowledge-exploratory", result.stderr)
            self.assertFalse(output.exists())

    def test_failed_worker_never_publishes_completed_summary(self):
        original, revised = fixture()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            args = [
                "paired", "--input", "prepared", "--output", str(output),
                "--run", "--acknowledge-exploratory",
            ]
            with (
                patch.object(sys, "argv", args),
                patch.object(paired, "output_path", return_value=output),
                patch.object(paired, "load_inputs", return_value=(
                    {"scope": "full_explicit"}, {"v2": [original], "v3": [revised]}, {}, [], {},
                )),
                patch.object(paired, "prepare_run", side_effect=lambda *args: output.mkdir()),
                patch.object(paired.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "worker")),
                redirect_stderr(StringIO()),
            ):
                self.assertEqual(paired.main(), 1)
            self.assertFalse((output / "summary.json").exists())


@unittest.skipUnless(importlib.util.find_spec("scipy"), "Paired statistics require scientific dependencies.")
class PairedStatisticsTests(unittest.TestCase):
    def test_error_delta_direction_and_identical_predictions(self):
        result = paired.paired_statistics([1, 2, 3], [0, 0, 0], [1, 2, 3], ["a", "b", "c"], 100)
        self.assertLess(result["delta_log_RMSE"], 0)
        self.assertLess(result["delta_log_RMSE_95"][1], 0)
        identical = paired.paired_statistics([1, 2], [0, 3], [0, 3], ["a", "b"], 50)
        self.assertEqual(identical["delta_log_RMSE_95"], [0, 0])
        self.assertEqual(identical["delta_log_MAE_95"], [0, 0])

    def test_group_bootstrap_keeps_whole_campaigns_and_is_deterministic(self):
        result = paired.paired_statistics([0, 0], [1, 3], [0, 0], ["same", "same"], 100)
        self.assertEqual(result["delta_log_RMSE_95"], [-math.sqrt(5), -math.sqrt(5)])
        self.assertEqual(result["v2_Spearman"], None)
        self.assertEqual(result, paired.paired_statistics([0, 0], [1, 3], [0, 0], ["same", "same"], 100))

    def test_nonfinite_or_misaligned_inputs_rejected(self):
        for target, left, right, groups in (
            ([], [], [], []), ([1], [float("nan")], [2], ["a"]),
            ([1], [2], [float("inf")], ["a"]), ([1], [2, 3], [2], ["a"]),
        ):
            with self.subTest(target=target), self.assertRaises(ValueError):
                paired.paired_statistics(target, left, right, groups, 10)

    def test_comparison_outputs_and_baseline_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "common").mkdir()
            expected = [
                {"video_id": identity, "group": identity, "views": str(views), "split": "test"}
                for identity, views in (("a", 10), ("b", 20))
            ]
            write_csv(output / "common" / "splits.csv", expected, list(expected[0]))
            fold_fields = ["fold", "video_id", "group", "role"]
            write_csv(output / "common" / "inner_folds.csv", [], fold_fields)
            for policy in paired.POLICIES:
                workspace = output / policy / "workspace"
                folder = workspace / "artifacts" / "v2"
                folder.mkdir(parents=True)
                write_json(workspace / "worker_complete.json", {"policy": policy})
                write_csv(folder / "inner_folds.csv", [], fold_fields)
                predictions = [
                    {**row, **{name + "_log": math.log1p(int(row["views"])) for name in paired.MODEL_NAMES}}
                    for row in expected
                ]
                write_csv(folder / "test_predictions.csv", predictions, list(predictions[0]))
                seed_rows = [
                    {"video_id": row["video_id"], "group": row["group"],
                     "target_log": math.log1p(int(row["views"])), "model": name,
                     "seed": seed, "prediction_log": math.log1p(int(row["views"]))}
                    for name in paired.GNN_MODELS for seed in paired.SEEDS for row in expected
                ]
                write_csv(folder / "gnn_seed_predictions.csv", seed_rows, list(seed_rows[0]))
            self.assertEqual(paired.compare_results(output), "full_retrained_paired")
            comparison = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(len(comparison["model_comparisons_seed42"]), 10)
            self.assertEqual(len(comparison["gnn_seed_comparisons"]["Subgraph structural"]), 4)
            self.assertTrue((output / "comparison.csv").exists())
            predictions[0]["Metadata Ridge_log"] = 0
            write_csv(folder / "test_predictions.csv", predictions, list(predictions[0]))
            with self.assertRaisesRegex(ValueError, "baseline predictions differ"):
                paired.compare_results(output)


class RealPreparedInputTests(unittest.TestCase):
    def test_actual_cohort_policy_separation_and_read_only_plan(self):
        prepared = ROOT / "artifacts" / "isolated_revision" / "final-v4"
        if not (prepared / "manifest.json").exists():
            self.skipTest("Local full-app-v3 data is not part of a fresh source checkout.")
        audit, rows, graphs, pairs, _ = paired.load_inputs(prepared)
        self.assertEqual(len(rows["v2"]), audit["split_counts"]["train"] + audit["split_counts"]["test"])
        self.assertEqual(
            [(r["video_id"], r["group"], r["split"]) for r in rows["v2"]],
            [(r["video_id"], r["group"], r["split"]) for r in rows["v3"]],
        )
        self.assertTrue(any(a["hook"] != b["hook"] for a, b in zip(rows["v2"], rows["v3"])))
        self.assertNotEqual(graphs["v2"], graphs["v3"])
        identities = {r["video_id"] for r in rows["v3"]}
        self.assertTrue(all(a in identities and b in identities for a, b in pairs))
        with tempfile.TemporaryDirectory() as directory:
            output = ROOT / "artifacts" / "paired_revision" / ("plan-" + Path(directory).name)
            result = subprocess.run([
                sys.executable, str(ROOT / "Scripts" / "run_paired_experiment.py"),
                "--input", str(prepared), "--output", str(output),
            ], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(result.stdout)
            self.assertEqual(plan["mode"], "plan_only")
            self.assertEqual(len(plan["models_per_policy"]), 10)
            self.assertFalse(plan["training_started"])
            self.assertFalse(Path(plan["output"]).exists())

    def test_actual_smoke_publishes_only_data_path_results(self):
        prepared = ROOT / "artifacts" / "isolated_revision" / "final-v4"
        if not (prepared / "manifest.json").exists():
            self.skipTest("Local full-app-v3 data is not part of a fresh source checkout.")
        base = ROOT / "artifacts" / "paired_revision"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="test-smoke-", dir=base) as directory:
            output = Path(directory) / "smoke"
            command = [
                sys.executable, str(ROOT / "Scripts" / "run_paired_experiment.py"),
                "--input", str(prepared), "--output", str(output), "--smoke",
            ]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["mode"], "smoke_mean_only")
            self.assertFalse((output / "comparison.json").exists())
            self.assertFalse((output / "v2" / "workspace" / "artifacts" / "v2").exists())
            self.assertFalse((output / "common" / "inner_folds.csv").exists())
            repeated = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("already exists", repeated.stderr)


if __name__ == "__main__":
    unittest.main()
