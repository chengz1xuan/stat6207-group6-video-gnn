"""Discover only completed, hash-verified local experiment bundles."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_FILES = (
    "primary_gnn.pt", "web_results.json", "optimization.json", "metrics.json",
    "test_predictions.csv", "oof_predictions.csv", "uncertainty.json",
    "cross_validation.json", "splits.csv", "seed_sensitivity.json",
    "error_slices.json", "inference_fixture.json",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_bundle(folder, root=ROOT):
    folder, root = Path(folder).resolve(), Path(root).resolve()
    if folder == root / "artifacts" / "v2":
        for name in ("primary_gnn.pt", "web_results.json", "optimization.json", "metrics.json"):
            if not (folder / name).is_file():
                raise ValueError(f"Original result file is missing: {folder / name}")
        return {"kind": "legacy", "label": "原版冻结实验"}
    allowed = root / "artifacts" / "isolated_revision"
    if allowed not in folder.parents:
        raise ValueError("An experiment must be inside artifacts/isolated_revision.")
    manifest_path = folder / "app_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "isolated_exploratory":
        raise ValueError(f"Unsupported App experiment manifest: {manifest_path}")
    expected_files = set(BUNDLE_FILES)
    if manifest.get("brand_transfer_completed"):
        expected_files.update(("leave_brand_out.csv", "leave_brand_out_summary.json"))
    if set(manifest["outputs"]) != expected_files:
        raise ValueError("The completed experiment has an incomplete output inventory.")
    for name in expected_files:
        if sha256(folder / name) != manifest["outputs"][name]:
            raise ValueError(f"Experiment output changed: {folder / name}")
    prepared = folder.parents[3]
    if folder != prepared / "evaluation" / "workspace" / "artifacts" / "v2":
        raise ValueError("Unexpected experiment workspace layout.")
    for name in ("manifest.json", "audit.json"):
        if sha256(prepared / name) != manifest["prepared"][name]:
            raise ValueError(f"Prepared experiment changed: {name}")
    for name in ("summary.json", "provenance.json"):
        if sha256(prepared / "evaluation" / name) != manifest["evaluation"][name]:
            raise ValueError(f"Evaluation record changed: {name}")
    prepared_manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected in prepared_manifest["outputs"].items():
        path = (prepared / relative).resolve()
        if prepared not in path.parents or sha256(path) != expected:
            raise ValueError(f"Prepared output changed: {relative}")
    provenance = json.loads((prepared / "evaluation" / "provenance.json").read_text(encoding="utf-8"))
    for name in ("pipeline.py", "revision.py"):
        if sha256(root / "src" / name) != provenance["training_sources"][name]:
            raise ValueError(f"Inference implementation differs from training: {name}")
    if sha256(root / "Scripts" / "text_policy.py") != manifest["text_policy_sha256"]:
        raise ValueError("Inference text policy differs from the completed training policy.")
    summary = json.loads((prepared / "evaluation" / "summary.json").read_text(encoding="utf-8"))
    if summary["mode"] != "full_retrained_exploratory":
        raise ValueError("Only completed full exploratory evaluations can be opened in the App.")
    return manifest


def available_experiments(root=ROOT, errors=None):
    root = Path(root).resolve()
    original = root / "artifacts" / "v2"
    validate_bundle(original, root)
    experiments = [(str(original), "原版冻结实验")]
    for path in sorted((root / "artifacts" / "isolated_revision").glob(
        "*/evaluation/workspace/artifacts/v2/app_manifest.json"
    )):
        try:
            manifest = validate_bundle(path.parent, root)
        except (OSError, ValueError, KeyError, TypeError) as error:
            if errors is None:
                raise
            errors.append(f'{path.parent.relative_to(root)}: {error}')
            continue
        experiments.append((str(path.parent), manifest["label"]))
    return experiments


def publish_bundle(prepared, folder, brand_transfer_completed):
    """Write the discovery marker last, after all training and export steps succeed."""
    audit = json.loads((prepared / "audit.json").read_text(encoding="utf-8"))
    names = list(BUNDLE_FILES)
    if brand_transfer_completed:
        names += ["leave_brand_out.csv", "leave_brand_out_summary.json"]
    manifest = {
        "schema_version": 1, "kind": "isolated_exploratory",
        "label": f"探索性修正实验 · {prepared.name}",
        "training_ready": False,
        "brand_transfer_completed": brand_transfer_completed,
        "split_counts": audit["split_counts"],
        "prepared": {name: sha256(prepared / name) for name in ("manifest.json", "audit.json")},
        "evaluation": {
            name: sha256(prepared / "evaluation" / name)
            for name in ("summary.json", "provenance.json")
        },
        "text_policy_sha256": sha256(prepared / "evaluation" / "workspace" / "src" / "text_policy.py"),
        "outputs": {name: sha256(folder / name) for name in names},
    }
    (folder / "app_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    validate_bundle(folder)
