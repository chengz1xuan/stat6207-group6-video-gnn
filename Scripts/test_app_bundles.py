import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from experiment_results import BUNDLE_FILES, available_experiments, sha256, validate_bundle


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        legacy = self.root / "artifacts" / "v2"
        legacy.mkdir(parents=True)
        for name in ("primary_gnn.pt", "web_results.json", "optimization.json", "metrics.json"):
            (legacy / name).write_bytes(b"fixture")
        policy = self.root / "Scripts" / "text_policy.py"
        policy.parent.mkdir()
        policy.write_bytes(b"fixture-policy")
        self.prepared = self.root / "artifacts" / "isolated_revision" / "full"
        evaluation = self.prepared / "evaluation"
        self.folder = evaluation / "workspace" / "artifacts" / "v2"
        self.folder.mkdir(parents=True)
        for name in BUNDLE_FILES:
            (self.folder / name).write_bytes(b"fixture")
        (self.prepared / "audit.json").write_text("{}", encoding="utf-8")
        (self.prepared / "videos.csv").write_bytes(b"fixture-data")
        (self.prepared / "manifest.json").write_text(json.dumps({
            "outputs": {"videos.csv": sha256(self.prepared / "videos.csv")},
        }), encoding="utf-8")
        (evaluation / "summary.json").write_text(
            json.dumps({"mode": "full_retrained_exploratory"}), encoding="utf-8",
        )
        source = self.root / "src"
        source.mkdir()
        for name in ("pipeline.py", "revision.py"):
            (source / name).write_bytes(b"fixture-source")
        (evaluation / "provenance.json").write_text(json.dumps({
            "training_sources": {name: sha256(source / name) for name in ("pipeline.py", "revision.py")},
        }), encoding="utf-8")
        self.manifest = {
            "schema_version": 1, "kind": "isolated_exploratory", "label": "fixture",
            "brand_transfer_completed": False,
            "outputs": {name: sha256(self.folder / name) for name in BUNDLE_FILES},
            "prepared": {name: sha256(self.prepared / name) for name in ("manifest.json", "audit.json")},
            "evaluation": {
                name: sha256(evaluation / name) for name in ("summary.json", "provenance.json")
            },
            "text_policy_sha256": sha256(policy),
        }

    def finish_bundle(self):
        (self.folder / "app_manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_incomplete_training_is_not_discovered(self):
        self.assertEqual(len(available_experiments(self.root)), 1)

    def test_completed_bundle_and_legacy_are_both_discovered(self):
        self.finish_bundle()
        self.assertEqual(len(available_experiments(self.root)), 2)
        self.assertEqual(validate_bundle(self.folder, self.root)["kind"], "isolated_exploratory")

    def test_changed_checkpoint_is_rejected(self):
        self.finish_bundle()
        (self.folder / "primary_gnn.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "output changed"):
            validate_bundle(self.folder, self.root)

    def test_invalid_optional_bundle_is_reported_without_breaking_main(self):
        self.finish_bundle()
        (self.folder / "primary_gnn.pt").write_bytes(b"changed")
        errors = []
        experiments = available_experiments(self.root, errors=errors)
        self.assertEqual(len(experiments), 1)
        self.assertEqual(len(errors), 1)
        self.assertIn("output changed", errors[0])
        with self.assertRaisesRegex(ValueError, "output changed"):
            validate_bundle(self.folder, self.root)

    def test_changed_preparation_is_rejected(self):
        self.finish_bundle()
        (self.prepared / "audit.json").write_text('{"changed": true}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Prepared experiment changed"):
            validate_bundle(self.folder, self.root)

    def test_different_inference_policy_is_rejected(self):
        self.finish_bundle()
        (self.root / "Scripts" / "text_policy.py").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Inference text policy"):
            validate_bundle(self.folder, self.root)

    def test_incomplete_inventory_is_rejected(self):
        self.manifest["outputs"].pop("primary_gnn.pt")
        self.finish_bundle()
        with self.assertRaisesRegex(ValueError, "incomplete output inventory"):
            validate_bundle(self.folder, self.root)

    def test_changed_prepared_data_is_rejected(self):
        self.finish_bundle()
        (self.prepared / "videos.csv").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Prepared output changed"):
            validate_bundle(self.folder, self.root)

    def test_changed_model_implementation_is_rejected(self):
        self.finish_bundle()
        (self.root / "src" / "pipeline.py").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Inference implementation"):
            validate_bundle(self.folder, self.root)

    def test_arbitrary_checkpoint_location_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "inside artifacts"):
            validate_bundle(self.root / "untrusted", self.root)

    def test_workflow_requires_explicit_training_and_acknowledgement(self):
        command = [
            sys.executable, str(ROOT / "Scripts" / "run_experiment.py"),
            "--output", str(self.prepared / "not-created"),
        ]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(json.loads(result.stdout)["training_started"])
        rejected = subprocess.run(command + ["--run"], cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertFalse((self.prepared / "not-created").exists())


if __name__ == "__main__":
    unittest.main()
