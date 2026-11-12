"""Compare v2/v3 preprocessing on one audited cohort; default is read-only planning."""

import argparse
import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from importlib.metadata import version

from evaluate_revision import load_prepared, smoke_evaluation, stage_sources
from isolate_revision import ROOT, exact_count, keyed, load_json, read_csv, write_csv, write_json


POLICIES = ("v2", "v3")
GNN_MODELS = ("Subgraph structural", "Subgraph + similarity")
SEEDS = (42, 7, 21, 84)
MODEL_NAMES = (
    "Mean", "Brand mean", "Metadata Ridge", "Pooled Ridge", "Pooled MLP",
    *GNN_MODELS, "Node structural", "Node + similarity", "Node no-brand edges",
)
NOTE = (
    "Internal paired re-evaluation on an already inspected holdout, not an untouched "
    "external test. Differences estimate the combined text/graph-policy effect, "
    "not the effect of any single correction or a causal creative uplift."
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def feature_fields():
    tree = ast.parse((ROOT / "src" / "pipeline.py").read_text(encoding="utf-8"))
    fields = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"CAT", "TEXT"}:
                    fields[target.id] = ast.literal_eval(node.value)
    if set(fields) != {"CAT", "TEXT"}:
        raise ValueError("Cannot identify the training feature fields.")
    return fields["CAT"], fields["TEXT"]


def validate_graph(graph):
    nodes = keyed(graph["nodes"], "id")
    if not nodes or graph["nodes"][0]["type"] != "video":
        raise ValueError(f"Missing video root: {graph['video_id']}")
    if not graph["edges"] or any(
        edge["source"] not in nodes or edge["target"] not in nodes
        for edge in graph["edges"]
    ):
        raise ValueError(f"Empty or dangling graph: {graph['video_id']}")


def assemble_inputs(prepared_rows, prepared_graphs, legacy_rows, legacy_graphs, categories, texts):
    """Keep only treatment fields/graphs different; take all controls from the audited rows."""
    legacy = keyed(legacy_rows, "video_id")
    old_graphs = keyed(legacy_graphs, "video_id")
    keyed(prepared_rows, "video_id")
    rows = {policy: [] for policy in POLICIES}
    graphs = {policy: [] for policy in POLICIES}
    controls = categories + [
        "youtube_id", "views", "duration_seconds", "age_days", "captured_at",
    ]
    for revised in prepared_rows:
        identity = revised["video_id"]
        if revised["split"] == "quarantine":
            continue
        original = legacy[identity]
        for field in controls:
            if original[field] != revised[field]:
                raise ValueError(f"Control differs between policies: {identity}: {field}")
        exact_count(revised["views"])
        baseline = dict(revised)
        for field in texts:
            if revised["original_" + field] != original[field]:
                raise ValueError(f"Original text provenance differs: {identity}: {field}")
            baseline[field] = original[field]
        rows["v2"].append(baseline)
        rows["v3"].append(dict(revised))
        for policy, graph in (("v2", old_graphs[identity]), ("v3", prepared_graphs[identity])):
            validate_graph(graph)
            if graph["video_id"] != identity:
                raise ValueError(f"Graph identity mismatch: {identity}")
            graphs[policy].append(graph)
    if not rows["v2"]:
        raise ValueError("No eligible videos.")
    return rows, graphs


def output_path(path):
    path = path.resolve()
    allowed = (ROOT / "artifacts" / "paired_revision").resolve()
    if allowed not in path.parents:
        raise ValueError(f"Output must be a new child directory beneath {allowed}")
    if path.exists():
        raise ValueError(f"Output already exists; choose a fresh directory: {path}")
    return path


def load_inputs(directory):
    prepared, audit, revised_rows, revised_graphs, pairs = load_prepared(directory)
    categories, texts = feature_fields()
    old_rows_path = ROOT / "data" / "processed" / "videos_v2.csv"
    old_graphs_path = ROOT / "artifacts" / "v2" / "raw_graphs.json"
    rows, graphs = assemble_inputs(
        revised_rows, revised_graphs, read_csv(old_rows_path, []),
        load_json(old_graphs_path), categories, texts,
    )
    train_groups = {row["group"] for row in rows["v3"] if row["split"] == "train"}
    if len(train_groups) < 3:
        raise ValueError("Paired evaluation needs at least three eligible training campaign groups.")
    provenance = {
        "prepared_directory": str(prepared),
        "prepared_manifest_sha256": sha256(prepared / "manifest.json"),
        "legacy_rows_sha256": sha256(old_rows_path),
        "legacy_graphs_sha256": sha256(old_graphs_path),
    }
    return audit, rows, graphs, pairs, provenance


def stage_policy(output, policy):
    if policy == "v3":
        return stage_sources(output)
    if policy != "v2":
        raise ValueError(f"Unknown policy: {policy}")
    workspace = output / "workspace"
    source = workspace / "src"
    source.mkdir(parents=True)
    (workspace / "artifacts").mkdir()
    for name in ("pipeline.py", "revision.py"):
        shutil.copyfile(ROOT / "src" / name, source / name)
    return workspace


def prepare_run(output, rows, graphs, pairs, provenance, smoke):
    output.mkdir(parents=True, exist_ok=False)
    common = output / "common"
    common.mkdir()
    split_fields = ["video_id", "youtube_id", "brand_family", "group", "split", "views"]
    write_csv(common / "splits.csv", [
        {field: row[field] for field in split_fields} for row in rows["v3"]
    ], split_fields)
    write_json(common / "video_pairs.json", pairs)
    if not smoke:
        from sklearn.model_selection import GroupKFold

        train = [row for row in rows["v3"] if row["split"] == "train"]
        folds = GroupKFold(3).split(train, groups=[row["group"] for row in train])
        fold_rows = [
            {"fold": fold, "video_id": train[int(i)]["video_id"],
             "group": train[int(i)]["group"], "role": role}
            for fold, (fit, validation) in enumerate(folds)
            for role, indices in (("fit", fit), ("validation", validation))
            for i in indices
        ]
        write_csv(common / "inner_folds.csv", fold_rows, ["fold", "video_id", "group", "role"])
    for policy in POLICIES:
        folder = output / policy
        folder.mkdir()
        write_csv(folder / "videos.csv", rows[policy], list(rows[policy][0]))
        write_json(folder / "raw_graphs.json", graphs[policy])
        stage_policy(folder, policy)
    write_json(output / "protocol.json", {
        "schema_version": 1, "kind": "paired_policy_evaluation",
        "mode": "smoke_mean_only" if smoke else "full_retrained",
        "policies": {
            "v2": "Original fields, original response-word filter and frozen v2 subgraphs.",
            "v3": "Audited fields, quantified-response filter and conservative rebuilt subgraphs.",
        },
        "controls": (
            "Identical eligible videos, labels, metadata, corrected campaign groups, "
            "outer split, inner folds, video-node pairs, model definitions and search budget."
        ),
        "seeds": list(SEEDS), "models": list(MODEL_NAMES),
        "bootstrap_draws": 2000, "bootstrap_seed": 44,
        "brand_transfer": "Not repeated; this experiment isolates the fixed grouped holdout.",
        "app_publication": False, "note": NOTE, "inputs": provenance,
        "original_sources": {
            name: sha256(ROOT / "src" / name) for name in ("pipeline.py", "revision.py")
        },
        "runner_sha256": sha256(Path(__file__)),
        "files": {
            str(path.relative_to(output)): sha256(path)
            for path in sorted(output.rglob("*")) if path.is_file()
        },
    })


def validate_run(output):
    protocol = load_json(output / "protocol.json")
    if protocol["kind"] != "paired_policy_evaluation":
        raise ValueError("Not a paired experiment.")
    if protocol["runner_sha256"] != sha256(Path(__file__)):
        raise ValueError("Paired runner changed after the experiment was prepared.")
    for relative, expected in protocol["files"].items():
        path = (output / relative).resolve()
        if output not in path.parents or sha256(path) != expected:
            raise ValueError(f"Paired input/source changed: {relative}")
    return protocol


def train_worker(output, policy):
    protocol = validate_run(output)
    if protocol["mode"] != "full_retrained":
        raise ValueError("A smoke run cannot launch training.")
    workspace = output / policy / "workspace"
    if (workspace / "artifacts" / "v2").exists():
        raise ValueError(f"Worker results already exist; do not overwrite or resume: {policy}")
    os.environ["MPLCONFIGDIR"] = str(workspace / "tmp" / "matplotlib")
    sys.path.insert(0, str(ROOT / ".project-deps"))
    sys.path.insert(0, str(workspace / "src"))
    if policy == "v3":
        # Preparation imports the live policy; training must use the frozen copy.
        sys.modules.pop("text_policy", None)
        import text_policy

        if Path(text_policy.__file__).resolve().parent != workspace / "src":
            raise ValueError("Training text policy did not resolve to the isolated source.")
    import numpy as np
    import pandas as pd
    import revision

    if Path(revision.__file__).resolve().parent != workspace / "src":
        raise ValueError("Training did not resolve to the isolated source.")
    df = pd.read_csv(output / policy / "videos.csv")
    for field in revision.CAT + revision.TEXT:
        df[field] = df[field].fillna("unknown" if field in revision.CAT else "")
    graphs = load_json(output / policy / "raw_graphs.json")
    if df.video_id.tolist() != [graph["video_id"] for graph in graphs]:
        raise ValueError("Training graph order differs from video order.")
    train = np.flatnonzero((df["split"] == "train").to_numpy())
    test = np.flatnonzero((df["split"] == "test").to_numpy())
    pairs = {tuple(pair) for pair in load_json(output / "common" / "video_pairs.json")}
    features, models, configs, predictions, _, _, _, _ = revision.train_all(
        df, graphs, train, test, pairs,
    )
    expected_folds = read_csv(output / "common" / "inner_folds.csv", [])
    if read_csv(workspace / revision.OUT / "inner_folds.csv", []) != expected_folds:
        raise ValueError("Training used different inner folds.")
    y = np.log1p(df.views.to_numpy(float))
    seed_rows = []
    for name in GNN_MODELS:
        training_graphs = features.transform(
            df.iloc[train], [graphs[i] for i in train], name == "Subgraph + similarity",
        )
        test_graphs = features.transform(
            df.iloc[test], [graphs[i] for i in test], name == "Subgraph + similarity",
        )
        for seed in SEEDS:
            if seed == 42:
                logs = predictions[name]
                model = models[name]
                config = configs[name]
            else:
                model, config = revision.fit_neural(
                    name, training_graphs, y[train], wd=configs[name]["weight_decay"],
                    epochs=configs[name]["epochs"], seed=seed,
                )
                logs = revision.predict(model, config, test_graphs)
            # Keep each fitted GNN, not just predictions, for reproducibility.
            revision.torch.save({
                "state_dict": model.state_dict(), "features": features,
                "config": config, "model_name": name, "seed": seed,
                "train_ids": df.iloc[train].video_id.tolist(),
                "test_ids": df.iloc[test].video_id.tolist(),
            }, revision.OUT / f'gnn_{"similarity" if "similarity" in name else "structural"}_seed{seed}.pt')
            for index, log in zip(test, logs):
                row = df.iloc[int(index)]
                seed_rows.append({
                    "video_id": row.video_id, "group": row.group, "target_log": float(y[index]),
                    "model": name, "seed": seed, "prediction_log": float(log),
                })
    write_csv(workspace / revision.OUT / "gnn_seed_predictions.csv", seed_rows, list(seed_rows[0]))
    write_json(workspace / "worker_complete.json", {
        "policy": policy, "train_size": len(train), "test_size": len(test),
        "environment": {name: version(name) for name in (
            "numpy", "pandas", "scipy", "scikit-learn", "torch", "torch-geometric",
        )},
    })


def align_predictions(left, right, expected):
    left, right, expected = (keyed(records, "video_id") for records in (left, right, expected))
    if set(left) != set(right) or set(left) != set(expected):
        raise ValueError("Paired test prediction inventories differ.")
    identities = list(expected)
    for identity in identities:
        for field in ("group", "views"):
            if left[identity][field] != right[identity][field] or (
                left[identity][field] != expected[identity][field]
            ):
                raise ValueError(f"Paired prediction target/group differs: {identity}: {field}")
    return [left[key] for key in identities], [right[key] for key in identities]


def paired_statistics(target, left, right, groups, draws=2000):
    import numpy as np
    from scipy.stats import spearmanr

    target, left, right = (np.asarray(values, dtype=float) for values in (target, left, right))
    groups = np.asarray(groups)
    if not (len(target) == len(left) == len(right) == len(groups)) or not len(target):
        raise ValueError("Paired metric arrays must be nonempty and aligned.")
    if not all(np.isfinite(values).all() for values in (target, left, right)):
        raise ValueError("Nonfinite target or prediction in paired evaluation.")
    if draws < 1:
        raise ValueError("Bootstrap draws must be positive.")
    a, b = left - target, right - target
    unique = np.unique(groups)
    rng = np.random.default_rng(44)
    indices = [np.flatnonzero(groups == group) for group in unique]
    deltas = []
    for _ in range(draws):
        sample = np.concatenate([indices[int(i)] for i in rng.integers(len(unique), size=len(unique))])
        deltas.append([
            float(np.sqrt(np.mean(b[sample] ** 2)) - np.sqrt(np.mean(a[sample] ** 2))),
            float(np.mean(abs(b[sample])) - np.mean(abs(a[sample]))),
        ])
    bounds = np.percentile(deltas, [2.5, 97.5], axis=0)

    def correlation(values):
        if len(values) < 2 or np.ptp(values) == 0 or np.ptp(target) == 0:
            return None
        return float(spearmanr(target, values).statistic)

    return {
        "n": len(target), "campaign_groups": len(unique),
        "v2_log_RMSE": float(np.sqrt(np.mean(a ** 2))),
        "v3_log_RMSE": float(np.sqrt(np.mean(b ** 2))),
        "delta_log_RMSE": float(np.sqrt(np.mean(b ** 2)) - np.sqrt(np.mean(a ** 2))),
        "delta_log_RMSE_95": bounds[:, 0].tolist(),
        "v2_log_MAE": float(np.mean(abs(a))), "v3_log_MAE": float(np.mean(abs(b))),
        "delta_log_MAE": float(np.mean(abs(b)) - np.mean(abs(a))),
        "delta_log_MAE_95": bounds[:, 1].tolist(),
        "v2_Spearman": correlation(left), "v3_Spearman": correlation(right),
    }


def compare_results(output):
    import numpy as np

    expected = [
        row for row in read_csv(output / "common" / "splits.csv", []) if row["split"] == "test"
    ]
    folders = {policy: output / policy / "workspace" / "artifacts" / "v2" for policy in POLICIES}
    for policy in POLICIES:
        load_json(folders[policy].parents[1] / "worker_complete.json")
        if read_csv(folders[policy] / "inner_folds.csv", []) != read_csv(
            output / "common" / "inner_folds.csv", [],
        ):
            raise ValueError(f"Inner folds differ: {policy}")
    left, right = align_predictions(
        read_csv(folders["v2"] / "test_predictions.csv", []),
        read_csv(folders["v3"] / "test_predictions.csv", []), expected,
    )
    for name in ("Mean", "Brand mean", "Metadata Ridge"):
        if not np.allclose(
            [float(row[name + "_log"]) for row in left],
            [float(row[name + "_log"]) for row in right], rtol=0, atol=1e-6,
        ):
            raise ValueError(f"Shared-control baseline predictions differ: {name}")
    target = [math.log1p(exact_count(row["views"])) for row in left]
    groups = [row["group"] for row in left]
    comparisons = {
        name: paired_statistics(
            target, [row[name + "_log"] for row in left],
            [row[name + "_log"] for row in right], groups,
        ) for name in MODEL_NAMES
    }
    seed_predictions = {
        policy: read_csv(folder / "gnn_seed_predictions.csv", [])
        for policy, folder in folders.items()
    }
    seed_results = {}
    for name in GNN_MODELS:
        seed_results[name] = {}
        for seed in SEEDS:
            records = {}
            for policy in POLICIES:
                selected = [
                    row for row in seed_predictions[policy]
                    if row["model"] == name and int(row["seed"]) == seed
                ]
                records[policy] = keyed(selected, "video_id")
                if set(records[policy]) != {row["video_id"] for row in left}:
                    raise ValueError(f"Seed prediction coverage differs: {policy}: {name}: {seed}")
                for row, y in zip(left, target):
                    prediction = records[policy][row["video_id"]]
                    if prediction["group"] != row["group"] or not math.isclose(
                        float(prediction["target_log"]), y, rel_tol=0, abs_tol=1e-10,
                    ):
                        raise ValueError("Seed prediction target/group differs.")
            seed_results[name][str(seed)] = paired_statistics(
                target,
                [records["v2"][row["video_id"]]["prediction_log"] for row in left],
                [records["v3"][row["video_id"]]["prediction_log"] for row in left], groups,
            )
    report = {
        "mode": "full_retrained_paired", "delta_definition": "v3 minus v2; negative error is better",
        "bootstrap": "2000 paired campaign-group draws, seed 44; conditional on fixed split/fits.",
        "model_comparisons_seed42": comparisons, "gnn_seed_comparisons": seed_results,
        "shared_control_baselines_match": True, "note": NOTE,
    }
    write_json(output / "comparison.json", report)
    table = []
    for name, result in comparisons.items():
        record = {"model": name}
        for key, value in result.items():
            if isinstance(value, list):
                record[key + "_low"], record[key + "_high"] = value
            else:
                record[key] = value
        table.append(record)
    write_csv(output / "comparison.csv", table, list(table[0]))
    return report["mode"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Audited full prepared v3 directory.")
    parser.add_argument("--output", type=Path, required=True, help="Fresh artifacts\\paired_revision child.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Retrain both ten-model suites.")
    mode.add_argument("--smoke", action="store_true", help="Check shared data using only a training mean.")
    parser.add_argument("--acknowledge-exploratory", action="store_true")
    parser.add_argument("--worker", choices=POLICIES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.worker:
            if not args.run or not args.acknowledge_exploratory:
                raise ValueError("Worker training requires explicit execution and acknowledgement.")
            output = args.output.resolve()
            allowed = (ROOT / "artifacts" / "paired_revision").resolve()
            if allowed not in output.parents:
                raise ValueError("Worker output must be an isolated paired experiment.")
            train_worker(output, args.worker)
            return 0
        if args.run and not args.acknowledge_exploratory:
            raise ValueError("--run requires --acknowledge-exploratory.")
        output = output_path(args.output)
        audit, rows, graphs, pairs, provenance = load_inputs(args.input)
        if audit["scope"] != "full_explicit":
            raise ValueError("Paired experiments require a full prepared cohort, not a sample export.")
        if not args.run and not args.smoke:
            print(json.dumps({
                "mode": "plan_only", "training_started": False,
                "split_counts": audit["split_counts"], "models_per_policy": list(MODEL_NAMES),
                "gnn_seeds": list(SEEDS), "output": str(output), "note": NOTE,
            }))
            return 0
        prepare_run(output, rows, graphs, pairs, provenance, args.smoke)
        if args.smoke:
            summaries = {policy: smoke_evaluation(rows[policy]) for policy in POLICIES}
            if summaries["v2"] != summaries["v3"]:
                raise ValueError("Shared training-mean smoke checks disagree.")
            write_json(output / "smoke.json", {
                "not_a_model_comparison": True, "policies": summaries, "note": NOTE,
            })
            result_mode = "smoke_mean_only"
        else:
            for policy in POLICIES:
                subprocess.run([
                    sys.executable, str(Path(__file__).resolve()), "--input", str(args.input.resolve()),
                    "--output", str(output), "--run", "--acknowledge-exploratory",
                    "--worker", policy,
                ], cwd=ROOT, check=True)
            validate_run(output)
            result_mode = compare_results(output)
        summary = {
            "mode": result_mode, "complete": True, "split_counts": audit["split_counts"],
            "note": NOTE, "outputs": {
                str(path.relative_to(output)): sha256(path)
                for path in sorted(output.rglob("*"))
                if path.is_file() and "__pycache__" not in path.parts and "tmp" not in path.parts
            },
        }
        write_json(output / "summary.json", summary)
        print(json.dumps({"mode": result_mode, "complete": True, "output": str(output)}))
        return 0
    except (ValueError, KeyError, OSError, csv.Error, ArithmeticError,
            ImportError, subprocess.CalledProcessError) as error:
        print(f"Paired experiment failed; no completed summary published: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
