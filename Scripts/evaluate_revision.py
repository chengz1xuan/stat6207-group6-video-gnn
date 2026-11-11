"""Validate a prepared revision; training requires --run and explicit acknowledgement."""

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import sys
from importlib.metadata import version
from pathlib import Path

from isolate_revision import ROOT, exact_count, keyed, load_json, read_csv, write_json


def load_prepared(directory):
    directory = directory.resolve()
    allowed = (ROOT / "artifacts/isolated_revision").resolve()
    if allowed not in directory.parents:
        raise ValueError(f"Input must be an isolated run beneath {allowed}")
    manifest = load_json(directory / "manifest.json")
    if manifest["schema_version"] != 2:
        raise ValueError("Re-export the data with the current isolation script.")
    for relative, expected in manifest["outputs"].items():
        path = (directory / relative).resolve()
        if directory not in path.parents:
            raise ValueError(f"Manifest path escapes the run directory: {relative}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Prepared data changed since export: {relative}")
    for name in ("text_policy.py", "graph_revision.py", "isolate_revision.py"):
        expected = manifest["scripts"].get(name)
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"{name} changed; re-export data before evaluation.")
    audit = load_json(directory / "audit.json")
    if audit["schema_version"] != 2:
        raise ValueError("Unsupported prepared audit schema.")
    rows = read_csv(directory / "videos.csv", [])
    splits = keyed(read_csv(directory / "proposed_splits.csv", []), "video_id")
    graphs = keyed(load_json(directory / "raw_graphs.json"), "video_id")
    records = keyed(rows, "video_id")
    keyed(rows, "youtube_id")
    if set(records) != set(splits) or set(records) != set(graphs):
        raise ValueError("Video, graph and split inventories disagree.")
    partitions = {}
    for row in rows:
        split = splits[row["video_id"]]
        for field in ("youtube_id", "group", "campaign_id", "split"):
            if split[field] != row[field]:
                raise ValueError(f"Prepared row/split mismatch: {row['video_id']}: {field}")
        if row["split"] not in {"train", "test", "quarantine"}:
            raise ValueError(f"Invalid partition: {row['split']}")
        partitions.setdefault(row["group"], set()).add(row["split"])
        exact_count(row["views"])
        graph = graphs[row["video_id"]]
        nodes = keyed(graph["nodes"], "id")
        if any(edge["source"] not in nodes or edge["target"] not in nodes for edge in graph["edges"]):
            raise ValueError(f"Dangling prepared graph edge: {row['video_id']}")
    if any(len(partition) != 1 for partition in partitions.values()):
        raise ValueError("A campaign crosses partitions, including quarantine.")
    train = [row for row in rows if row["split"] == "train"]
    test = [row for row in rows if row["split"] == "test"]
    if not train or not test:
        raise ValueError("Both eligible train and test partitions are required.")
    pairs = load_json(directory / "video_pairs.json")
    eligible = {row["video_id"] for row in train + test}
    if any(a == b or a not in eligible or b not in eligible for a, b in pairs):
        raise ValueError("Video relation includes an ineligible/quarantined reference.")
    return directory, audit, rows, graphs, pairs


def replace_function(source, name, replacement, class_name=None):
    tree = ast.parse(source)
    container = tree
    if class_name:
        container = next(node for node in tree.body
                         if isinstance(node, ast.ClassDef) and node.name == class_name)
    matches = [node for node in container.body
               if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise ValueError(f"Expected one source function {class_name or ''}.{name}")
    node = matches[0]
    prefix = " " * node.col_offset
    lines = source.splitlines(keepends=True)
    lines[node.lineno - 1:node.end_lineno] = [
        prefix + line + "\n" for line in replacement.splitlines()
    ]
    result = "".join(lines)
    ast.parse(result)
    return result


def stage_sources(output):
    workspace = output / "workspace"
    src = workspace / "src"
    src.mkdir(parents=True)
    (workspace / "artifacts").mkdir()
    module = (ROOT / "src/pipeline.py").read_text(encoding="utf-8")
    module = replace_function(
        module, "text",
        "def text(self, d):\n"
        " return [' '.join(response_free_text(r.get(c, '')) for c in TEXT) for _, r in d.iterrows()]",
        class_name="Features",
    )
    (src / "pipeline.py").write_text(
        "from text_policy import response_free_text\n" + module, encoding="utf-8",
    )
    module = (ROOT / "src/revision.py").read_text(encoding="utf-8")
    module = replace_function(
        module, "safe_text",
        "def safe_text(v):\n return response_free_text(v)",
    )
    (src / "revision.py").write_text(
        "from text_policy import response_free_text\n" + module, encoding="utf-8",
    )
    (src / "text_policy.py").write_bytes(Path(__file__).with_name("text_policy.py").read_bytes())
    return workspace


def smoke_evaluation(rows):
    train = [math.log1p(exact_count(row["views"])) for row in rows if row["split"] == "train"]
    test = [math.log1p(exact_count(row["views"])) for row in rows if row["split"] == "test"]
    prediction = sum(train) / len(train)
    return {
        "mode": "smoke_mean_only", "not_a_gnn_evaluation": True,
        "train_size": len(train), "test_size": len(test),
        "mean_prediction_log": prediction,
        "log_MAE": sum(abs(value - prediction) for value in test) / len(test),
        "log_RMSE": math.sqrt(sum((value - prediction) ** 2 for value in test) / len(test)),
        "note": "Checks the revised data/split path only; not evidence of model improvement.",
    }


def export_app_results(revision, df, gs, tr, te, audit, trained, transfer):
    features, models, configs, predictions, primary, bands, _, uncertainty = trained
    cases, brief, _ = revision.scenarios(
        df, gs, tr, te, features, models[primary], configs[primary], primary, bands[primary],
    )
    import numpy as np

    slices = []
    observed = np.log1p(df.iloc[te].views.to_numpy(float))
    for brand in sorted(df.iloc[te].brand_family.unique()):
        mask = (df.iloc[te].brand_family == brand).to_numpy()
        slices.append({
            "slice": brand, "n": int(mask.sum()),
            **revision.metric(observed[mask], predictions[primary][mask]),
        })
    revision.dump(revision.OUT / "error_slices.json", slices)
    split = {
        "train": len(tr), "test": len(te), "quarantine": audit["split_counts"]["quarantine"],
        "selection_rule": audit["split_policy"], "seed": 42,
    }
    revision.dump(revision.OUT / "split_policy.json", split)
    graph_audit = {
        "graphs": len(gs), "nodes": sum(len(g["nodes"]) for g in gs),
        "provided_edges": sum(len(g["edges"]) for g in gs),
        "policy": audit["graph_policy"],
    }
    revision.dump(revision.OUT / "graph_audit.json", graph_audit)
    videos = json.loads(df.to_json(orient="records", force_ascii=False))
    revision.dump(revision.OUT / "web_results.json", {
        "experiment_kind": "isolated_exploratory", "training_ready": False,
        "audit": audit, "metrics": load_json(revision.OUT / "metrics.json"),
        "split": split, "uncertainty": uncertainty, "lobo": transfer,
        "videos": videos, "graphs": gs, "cases": cases, "new_video": brief,
        "slices": slices, "graph_audit": graph_audit,
    })
    data = features.transform(df, gs, primary == "Subgraph + similarity")
    logs = revision.predict(models[primary], configs[primary], data)
    revision.dump(revision.OUT / "inference_fixture.json", [
        {"video_id": row["video_id"], "expected_log": float(log)}
        for row, log in zip(videos, logs)
    ])


def train_evaluate(directory, output, rows, graphs, pairs, skip_brand_transfer):
    train_groups = {row["group"] for row in rows if row["split"] == "train"}
    if len(train_groups) < 3:
        raise ValueError("Full evaluation needs at least three eligible training campaign groups.")
    workspace = stage_sources(output)
    os.environ["MPLCONFIGDIR"] = str(workspace / "tmp/matplotlib")
    sys.path.insert(0, str(ROOT / ".project-deps"))
    sys.path.insert(0, str(workspace / "src"))
    # These imports resolve to the isolated copies; the original modules never execute.
    import revision
    import numpy as np
    import pandas as pd

    if Path(revision.__file__).resolve().parent != (workspace / "src").resolve():
        raise ValueError("Training module did not resolve to the isolated source copy.")
    df = pd.read_csv(directory / "videos.csv")
    df = df[df["split"] != "quarantine"].reset_index(drop=True)
    for field in revision.CAT + revision.TEXT:
        df[field] = df[field].fillna("unknown" if field in revision.CAT else "")
    gs = [graphs[identity] for identity in df.video_id]
    tr = np.flatnonzero((df["split"] == "train").to_numpy())
    te = np.flatnonzero((df["split"] == "test").to_numpy())
    pairs = {tuple(pair) for pair in pairs}
    df[["video_id", "youtube_id", "group", "brand_family", "split"]].to_csv(
        revision.OUT / "splits.csv", index=False,
    )
    trained = revision.train_all(df, gs, tr, te, pairs)
    _, _, _, _, primary, _, results, _ = trained
    transfer = None if skip_brand_transfer else revision.leave_brand_out(df, gs, pairs)
    metrics = load_json(revision.OUT / "metrics.json")
    if metrics["train_size"] != len(tr) or metrics["test_size"] != len(te):
        raise ValueError("Evaluated split size does not match the prepared partition.")
    predictions = pd.read_csv(revision.OUT / "test_predictions.csv")
    if set(predictions.video_id) != set(df.iloc[te].video_id):
        raise ValueError("Test predictions do not cover the prepared held-out videos.")
    export_app_results(
        revision, df, gs, tr, te, load_json(directory / "audit.json"), trained, transfer,
    )
    return {
        "mode": "full_retrained_exploratory",
        "train_size": len(tr), "test_size": len(te), "primary_gnn": primary,
        "metrics": results, "brand_transfer": transfer,
        "results_path": str((workspace / revision.OUT).resolve()),
        "note": "New cohort/graph policy/split. Not a paired improvement over frozen v2 scores.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Prepared isolated run directory.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Refit all eight original model definitions.")
    mode.add_argument("--smoke", action="store_true", help="Small data-path check using a train-only mean.")
    parser.add_argument("--acknowledge-exploratory", action="store_true",
                        help="Acknowledge unresolved manual reviews and the non-production status.")
    parser.add_argument("--skip-brand-transfer", action="store_true")
    args = parser.parse_args()
    try:
        directory, audit, rows, graphs, pairs = load_prepared(args.input)
    except (ValueError, KeyError, OSError, csv.Error, ArithmeticError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 1
    return evaluate_args(args, directory, audit, rows, graphs, pairs)


def evaluate_args(args, directory, audit, rows, graphs, pairs):
    try:
        if not args.run and not args.smoke:
            print(json.dumps({
                "mode": "plan_only", "scope": audit["scope"],
                "split_counts": audit["split_counts"],
                "training_ready": audit["training_ready"],
                "models": ["Mean", "Metadata Ridge", "Pooled Ridge", "Pooled MLP",
                           "Subgraph structural", "Subgraph + similarity",
                           "Node structural", "Node + similarity"],
                "training_started": False,
            }, ensure_ascii=False))
            return 0
        if args.run and not args.acknowledge_exploratory:
            raise ValueError("Training requires --acknowledge-exploratory; manual reviews remain unresolved.")
        if args.run and audit["scope"] != "full_explicit":
            raise ValueError("Use --smoke for samples; full refitting requires a --full prepared cohort.")
        output = directory / ("evaluation" if args.run else "evaluation-smoke")
        output.mkdir(exist_ok=False)
        if args.smoke:
            summary = smoke_evaluation(rows)
        else:
            summary = train_evaluate(directory, output, rows, graphs, pairs, args.skip_brand_transfer)
        write_json(output / "summary.json", summary)
        write_json(output / "provenance.json", {
            "prepared_manifest_sha256": hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest(),
            "training_sources": {
                name: hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest()
                for name in ("pipeline.py", "revision.py")
            },
            "evaluation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "acknowledged_exploratory": args.acknowledge_exploratory,
            "environment": None if args.smoke else {
                name: version(name) for name in (
                    "numpy", "pandas", "scipy", "scikit-learn", "matplotlib",
                    "torch", "torch-geometric", "streamlit",
                )
            },
        })
        if args.run:
            sys.path.insert(0, str(ROOT / "src"))
            from experiment_results import publish_bundle
            publish_bundle(
                directory, output / "workspace" / "artifacts" / "v2",
                brand_transfer_completed=not args.skip_brand_transfer,
            )
        print(json.dumps(summary, ensure_ascii=False))
        return 0
    except (ValueError, KeyError, OSError, csv.Error, ArithmeticError, ImportError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
