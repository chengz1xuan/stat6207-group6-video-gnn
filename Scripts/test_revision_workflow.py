import ast
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from evaluate_revision import load_prepared, replace_function, smoke_evaluation, stage_sources
from isolate_revision import ROOT, campaign_groups, load_json, read_csv
from text_policy import response_free_text
from graph_revision import rebuild_graph


class TextPolicyTests(unittest.TestCase):
    def test_ordinary_like_share_and_decimal_survive(self):
        for text in (
            "Ingredients like fresh eggs and French butter.",
            "Share happiness and blessings with family.",
            "Use rewards like cash and get 1.5 times the value.",
            "Like our page to support the charity.",
            "Register at https://example.com/promo for $49.",
        ):
            self.assertEqual(response_free_text(text), text)

    def test_only_response_sentences_removed(self):
        self.assertEqual(
            response_free_text("Fresh eggs. The video received 1.2 million views. Share happiness."),
            "Fresh eggs. Share happiness.",
        )
        for text in ("Likes: 1,234.", "View count = 500.", "1,000views.", "Millions of views."):
            self.assertEqual(response_free_text(text), "")
        self.assertEqual(response_free_text("\u64ad\u653e\u91cf\uff1a120\u842c"), "")

    def test_json_and_nul_cleaning(self):
        self.assertEqual(response_free_text('"Fresh\\u0000 eggs."'), "Fresh eggs.")

    def test_real_maxim_creative_nodes_are_restored(self):
        rows = read_csv(ROOT / "data/processed/videos_v2.csv", [])
        row = dict(next(row for row in rows if row["youtube_id"] == "CxbnjaET7ew"))
        row["hook_status"] = "duplicate_summary"
        row["hook"] = ""
        folder = ROOT / "data_topic3" / row["package"]
        tables = {
            name: read_csv(folder / (name + ".csv"), [])
            for name in ("video_nodes", "aspect_nodes", "evidence_nodes", "candidate_edges")
        }
        graph, _ = rebuild_graph(row, tables, load_json(ROOT / "Scripts/review_rules.json"), [])
        fields = {node["field"]: node["text"] for node in graph["nodes"] if node["type"] == "aspect"}
        self.assertIn("like fresh eggs", fields["main_message"])
        self.assertIn("share happiness", fields["value_promise"])
        self.assertNotIn("hook", fields)
        self.assertNotIn("proof", fields)


class EvaluationTests(unittest.TestCase):
    def test_source_patches_are_local_and_syntactically_valid(self):
        before = {
            name: hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest()
            for name in ("pipeline.py", "revision.py")
        }
        with tempfile.TemporaryDirectory() as directory:
            workspace = stage_sources(Path(directory))
            for name in ("pipeline.py", "revision.py", "text_policy.py"):
                ast.parse((workspace / "src" / name).read_text(encoding="utf-8"))
            text = (workspace / "src/pipeline.py").read_text(encoding="utf-8")
            self.assertIn("response_free_text(r.get(c, ''))", text)
        self.assertEqual(before, {
            name: hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest()
            for name in before
        })

    def test_source_patch_rejects_missing_function(self):
        with self.assertRaises(ValueError):
            replace_function("x = 1\n", "missing", "def missing():\n pass")

    def test_smoke_mean_uses_training_labels_only(self):
        rows = [
            {"split": "train", "views": "10"},
            {"split": "test", "views": "100"},
            {"split": "quarantine", "views": "10000000"},
        ]
        result = smoke_evaluation(rows)
        rows[1]["views"] = "1000000"
        self.assertEqual(result["mean_prediction_log"], smoke_evaluation(rows)["mean_prediction_log"])
        self.assertEqual(result["train_size"], 1)

    def test_registry_merges_known_series_and_quarantines_pending(self):
        rows = read_csv(ROOT / "data/processed/videos_v2.csv", [])
        splits = {row["youtube_id"]: row for row in read_csv(ROOT / "artifacts/v2/splits.csv", [])}
        rules = load_json(ROOT / "Scripts/review_rules.json")
        groups, quarantine, _ = campaign_groups(rows, splits, rules)
        for family in rules["campaign_families"]:
            self.assertEqual(len({groups[identity] for identity in family["youtube_ids"]}), 1)
            if family["status"] == "needs_review":
                self.assertTrue(set(family["youtube_ids"]) <= quarantine)
        self.assertEqual(groups["jlB5oiVibQU"], groups["m8qVqW63VQs"])
        self.assertEqual(groups["8ySmQhxarpY"], groups["D1H6HkY-tCo"])

    def test_sample_export_plan_smoke_and_tamper_guard(self):
        base = ROOT / "artifacts/isolated_revision"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="test-", dir=base) as directory:
            output = Path(directory) / "prepared"
            command = [sys.executable, str(ROOT / "Scripts/isolate_revision.py"), "--output", str(output)]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            _, audit, rows, graphs, _ = load_prepared(output)
            self.assertEqual(len(rows), 12)
            self.assertEqual(audit["schema_version"], 2)
            self.assertEqual(set(graphs), {row["video_id"] for row in rows})
            self.assertTrue(all(not row["hook"] for row in rows))
            evaluator = [sys.executable, str(ROOT / "Scripts/evaluate_revision.py"), "--input", str(output)]
            result = subprocess.run(evaluator, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(json.loads(result.stdout)["training_started"])
            self.assertFalse((output / "evaluation").exists())
            blocked = subprocess.run(evaluator + ["--run"], cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(blocked.returncode, 0)
            self.assertFalse((output / "evaluation").exists())
            smoke = subprocess.run(evaluator + ["--smoke"], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(smoke.returncode, 0, smoke.stderr)
            summary = load_json(output / "evaluation-smoke/summary.json")
            self.assertTrue(summary["not_a_gnn_evaluation"])
            repeated = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(repeated.returncode, 0)
            path = output / "videos.csv"
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "changed since export"):
                load_prepared(output)


if __name__ == "__main__":
    unittest.main()
