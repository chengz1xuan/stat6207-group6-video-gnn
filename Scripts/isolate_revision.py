"""Offline, sample-first revision. Never import the training/collection pipeline."""

import argparse
import csv
import hashlib
import io
import json
import random
import re
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from graph_revision import rebuild_graph, video_pairs
from text_policy import response_free_text

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IDS = (
    "PhVN9idahMQ", "mYm7ef4MEqE", "rUxv3XvskeU",
    "0_1xI0ihZSk", "Ghs18rt-6pw",
    "N1XdXPMy0vQ", "X2dwXXFghCA", "Hki21yPVTS8", "demkAoN7Bzw",
    "WZTOLwgO93c", "XtXY6tEdn9g", "-5P9rd8IuV0",
)
def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_csv(path, issues):
    text = path.read_text(encoding="utf-8-sig")
    if "\x00" in text:
        issues.append({
            "kind": "nul_removed_in_copy", "source": str(path.relative_to(ROOT)),
            "count": text.count("\x00"),
            "scope": "entire input file scanned; only selected rows exported",
        })
    return list(csv.DictReader(io.StringIO(text.replace("\x00", ""))))


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def clean_value(value):
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, list):
        return [clean_value(item) for item in value]
    if isinstance(value, dict):
        result = {key: clean_value(item) for key, item in value.items()}
        if result.get("value_json"):
            result["value_json"] = json.dumps(
                clean_value(json.loads(result["value_json"])), ensure_ascii=False,
            )
        return result
    return value


def youtube_id(url):
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"Expected HTTPS YouTube URL: {url}")
    if parsed.hostname in {"www.youtube.com", "youtube.com", "m.youtube.com"}:
        values = parse_qs(parsed.query).get("v", [])
        result = values[0] if parsed.path == "/watch" and len(values) == 1 else ""
    elif parsed.hostname == "youtu.be":
        result = parsed.path.lstrip("/")
    else:
        result = ""
    if not re.fullmatch(r"[\w-]{11}", result, flags=re.ASCII):
        raise ValueError(f"Invalid YouTube video identity: {url}")
    return result


def exact_count(value):
    count = Decimal(str(value))
    if not count.is_finite() or count < 0 or count != count.to_integral_value():
        raise ValueError(f"Expected a nonnegative exact count, got {value!r}")
    return int(count)


def keyed(rows, field):
    result = {}
    for row in rows:
        key = row[field]
        if not key or key in result:
            raise ValueError(f"Missing/duplicate {field}: {key!r}")
        result[key] = row
    return result


def check_label(row, snapshot, evidence, original):
    identity = row["youtube_id"]
    for record in (row, snapshot, evidence, original):
        if record["video_id"].removeprefix("video:") != row["video_id"]:
            raise ValueError(f"Source video_id mismatch: {identity}")
        if youtube_id(record["source_url"]) != identity:
            raise ValueError(f"Source URL mismatch: {identity}")
    for record in (snapshot, evidence):
        if record["youtube_id"] != identity:
            raise ValueError(f"Snapshot identity mismatch: {identity}")
        if exact_count(record["views"]) != exact_count(row["views"]):
            raise ValueError(f"Frozen label mismatch: {identity}")
    raw_count = evidence["evidence"]["views"].get("originalViewCount")
    if raw_count is not None and exact_count(raw_count) != exact_count(row["views"]):
        raise ValueError(f"Extracted originalViewCount mismatch: {identity}")
    times = [
        row["captured_at"], row["retrieved_at_utc"],
        snapshot["retrieved_at_utc"], evidence["retrieved_at_utc"],
    ]
    parsed = [datetime.fromisoformat(value.replace("Z", "+00:00")) for value in times]
    if any(value.utcoffset() is None for value in parsed) or len(set(parsed)) != 1:
        raise ValueError(f"Capture timestamp mismatch: {identity}")


def campaign_groups(rows, splits, rules):
    by_id = keyed(rows, "youtube_id")
    parent = {identity: identity for identity in by_id}

    def find(identity):
        while parent[identity] != identity:
            parent[identity] = parent[parent[identity]]
            identity = parent[identity]
        return identity

    def union(identities):
        for identity in identities[1:]:
            parent[find(identity)] = find(identities[0])

    seen = {}
    for row in rows:
        identity = row["youtube_id"]
        if splits[identity]["group"] != row["group"]:
            raise ValueError(f"Original group mismatch: {identity}")
        # Preserve every existing component; aliases may only merge, never split it.
        for kind, value in (
            ("original", row["group"]),
            ("campaign", re.sub(r"[^\w]+", " ", row["product_campaign"].lower()).strip()),
            ("message", row["main_message"].strip().lower()),
        ):
            if not value:
                continue
            key = (row["brand_family"], kind, value)
            if key in seen:
                union([seen[key], identity])
            else:
                seen[key] = identity
    pending = set()
    applied = []
    for family in rules["campaign_families"]:
        status = family["status"]
        if status not in {"text_evidence", "needs_review"}:
            raise ValueError(f"Unknown campaign review status: {status}")
        members = [identity for identity in family["youtube_ids"] if identity in by_id]
        if not members:
            continue
        if len({by_id[identity]["brand_family"] for identity in members}) != 1:
            raise ValueError(f"Cross-brand campaign rule: {family['campaign_id']}")
        union(members)
        if status == "needs_review":
            pending.update(members)
        applied.append({
            **family, "selected_members": members,
            "original_splits": sorted({splits[identity]["split"] for identity in members}),
        })
    components = {}
    for identity in by_id:
        components.setdefault(find(identity), []).append(identity)
    groups, quarantine = {}, set()
    for members in components.values():
        digest = hashlib.sha256("\n".join(sorted(members)).encode()).hexdigest()[:16]
        family_ids = sorted({
            family["campaign_id"] for family in applied
            if set(family["selected_members"]).intersection(members)
        })
        group_id = family_ids[0] if len(family_ids) == 1 else "campaign-" + digest
        for identity in members:
            groups[identity] = group_id
        if pending.intersection(members):
            quarantine.update(members)
    return groups, quarantine, applied


def proposed_split(groups, quarantine, fraction=0.2, seed=42):
    components = {}
    for identity, group in groups.items():
        if identity not in quarantine:
            components.setdefault(group, []).append(identity)
    n = sum(map(len, components.values()))
    if len(components) < 2:
        raise ValueError("At least two non-quarantined campaign groups are required.")
    ordered = sorted(components)
    random.Random(seed).shuffle(ordered)
    # Subset-sum chooses the closest whole-group count, without reading any labels.
    reachable = {0: ()}
    for group in ordered:
        for count, selected in list(reachable.items()):
            new_count = count + len(components[group])
            if new_count not in reachable:
                reachable[new_count] = selected + (group,)
    target = round(n * fraction)
    candidates = [count for count in reachable if 0 < count < n]
    count = min(candidates, key=lambda count: (abs(count - target), count))
    test_groups = set(reachable[count])
    return {
        identity: (
            "quarantine" if identity in quarantine
            else "test" if group in test_groups else "train"
        )
        for identity, group in groups.items()
    }


def safe_output(root, output):
    output = output.resolve()
    allowed = (root / "artifacts" / "isolated_revision").resolve()
    if output == allowed or allowed not in output.parents:
        raise ValueError(f"Output must be a new child directory of {allowed}")
    if output.exists():
        raise ValueError(f"Output already exists; choose a fresh run directory: {output}")
    return output


def run(args):
    issues = []
    default_run = "full-v2" if args.full else "sample-v2"
    output = safe_output(ROOT, args.output or ROOT / "artifacts/isolated_revision" / default_run)
    rules = load_json(args.rules)
    all_rows = read_csv(ROOT / "data/processed/videos_v2.csv", issues)
    indexed = keyed(all_rows, "youtube_id")
    family_ids = [rule["campaign_id"] for rule in rules["campaign_families"]]
    if len(family_ids) != len(set(family_ids)):
        raise ValueError("Duplicate campaign IDs in the registry.")
    for family in rules["campaign_families"]:
        if not family["reason"] or not family["youtube_ids"]:
            raise ValueError(f"Missing campaign evidence: {family['campaign_id']}")
        missing = set(family["youtube_ids"]) - set(indexed)
        if missing:
            raise ValueError(f"Stale campaign members: {sorted(missing)}")
    selected_ids = sorted(indexed) if args.full else list(SAMPLE_IDS)
    missing = set(selected_ids) - set(indexed)
    if missing:
        raise ValueError(f"Missing requested videos: {sorted(missing)}")
    rows = [clean_value(indexed[identity]) for identity in selected_ids]
    splits = keyed(read_csv(ROOT / "artifacts/v2/splits.csv", issues), "youtube_id")
    snapshots = keyed(read_csv(ROOT / "data/youtube/snapshot.csv", issues), "youtube_id")
    global_groups, global_quarantine, families = campaign_groups(all_rows, splits, rules)
    groups = {identity: global_groups[identity] for identity in selected_ids}
    quarantine = global_quarantine.intersection(selected_ids)
    families = [family for family in families if set(family["selected_members"]).intersection(selected_ids)]
    input_files = {
        ROOT / "data/processed/videos_v2.csv", ROOT / "artifacts/v2/splits.csv",
        ROOT / "data/youtube/snapshot.csv", args.rules.resolve(),
    }
    source_tables = {}
    for package in sorted({row["package"] for row in rows}):
        folder = ROOT / "data_topic3" / package
        if folder.resolve().parent != (ROOT / "data_topic3").resolve():
            raise ValueError(f"Invalid source package path: {package}")
        source_tables[package] = {
            name: read_csv(folder / (name + ".csv"), issues)
            for name in ("video_nodes", "aspect_nodes", "evidence_nodes", "candidate_edges")
        }
        input_files.update(folder / (name + ".csv") for name in source_tables[package])
    revised_graphs, changes = [], []
    for row in rows:
        identity = row["youtube_id"]
        source = source_tables[row["package"]]
        originals = keyed([
            {**value, "video_id": value["video_id"].removeprefix("video:")}
            for value in source["video_nodes"]
        ], "video_id")
        check_label(
            row, snapshots[identity],
            load_json(ROOT / "data/youtube" / (identity + ".json")),
            originals[row["video_id"]],
        )
        split = splits[identity]
        if split["video_id"] != row["video_id"] or split["brand_family"] != row["brand_family"]:
            raise ValueError(f"Original split identity mismatch: {identity}")
        input_files.add(ROOT / "data/youtube" / (identity + ".json"))
        row["hook_status"] = (
            "duplicate_summary" if row["hook"].strip().lower() == row["main_message"].strip().lower()
            else "source_annotation"
        )
        for field in ("main_message", "hook", "value_promise", "cta"):
            row["original_" + field] = row[field]
            row[field] = response_free_text(row[field])
            if row[field] != row["original_" + field]:
                issues.append({
                    "kind": "text_cleaning", "youtube_id": identity, "field": field,
                    "before": row["original_" + field], "after": row[field],
                })
            if row["hook_status"] == "duplicate_summary" and field == "hook":
                row[field] = ""
        for field in ("likes", "comments"):
            row[field + "_archived"] = row[field]
            row[field] = row[field + "_youtube"]
        revised, removed = rebuild_graph(row, source, rules, issues)
        revised_graphs.append(revised)
        changes.extend(removed)
    assignments = proposed_split(groups, quarantine)
    for review in rules["creative_reviews"]:
        if review["youtube_id"] in selected_ids:
            issues.append({"kind": "manual_creative_review", **review})
    proposed = []
    for row in rows:
        identity = row["youtube_id"]
        row["original_group"] = row["group"]
        row["original_split"] = splits[identity]["split"]
        row["group"] = groups[identity]
        row["campaign_id"] = groups[identity]
        row["split"] = assignments[identity]
        proposed.append({
            key: row[key] for key in (
                "video_id", "youtube_id", "brand_family", "original_group",
                "original_split", "campaign_id", "group", "split",
            )
        })
    for group in set(groups.values()):
        if len({assignments[key] for key in groups if groups[key] == group}) != 1:
            raise ValueError(f"Campaign crosses proposed partitions: {group}")
    fields = list(rows[0])
    report = {
        "scope": "full_explicit" if args.full else "12_video_sample",
        "schema_version": 2,
        "training_ready": False,
        "limitations": [
            "No training, recollection, or update to existing results.",
            "Campaign rules cover reviewed examples, not exhaustive campaign lineage.",
            "Sample campaign IDs and split must not be joined into a full-cohort split.",
            "Graphs rebuilt from source nodes/edges, not re-extracted from video footage.",
            "Manual creative reviews remain unresolved; labels are not guessed.",
            "All eligible videos are repartitioned; existing checkpoints/metrics are incompatible.",
            "Heuristic supports and derived-summary evidence are excluded by a fixed conservative policy.",
            "Duplicate-summary hooks are blanked, not replaced with invented hooks.",
        ],
        "video_count": len(rows),
        "split_counts": {
            split: sum(value == split for value in assignments.values())
            for split in ("train", "test", "quarantine")
        },
        "split_policy": "Seed 42; closest rounded 20% of eligible videos, whole groups only; no label access.",
        "applied_campaign_families": families,
        "excluded_support_edges": changes,
        "graph_policy": {
            "evidence": "description, spoken dialogue, on-screen text, end card",
            "support": "source-provided deterministic observation links only; no new semantic claims",
            "caps": "three observations per kind; one aspect per field; four visual/audience signals",
            "model_comparison": "Different evidence policy and split; old metrics are not a paired baseline.",
        },
        "issues": issues,
        "rules_sha256": hashlib.sha256(args.rules.read_bytes()).hexdigest(),
    }
    # Validate everything before creating the isolated output directory.
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "videos.csv", rows, fields)
    write_csv(output / "proposed_splits.csv", proposed, list(proposed[0]))
    write_csv(output / "quarantined_videos.csv", [
        row for row in rows if row["split"] == "quarantine"
    ], fields)
    write_csv(output / "edge_changes.csv", changes, [
        "youtube_id", "source_edge_id", "action", "reason",
    ])
    write_json(output / "raw_graphs.json", revised_graphs)
    eligible_ids = {row["video_id"] for row in rows if row["split"] != "quarantine"}
    write_json(output / "video_pairs.json", video_pairs(source_tables, eligible_ids))
    write_json(output / "review_rules.json", rules)
    for package, tables in source_tables.items():
        chosen = {row["video_id"] for row in rows if row["package"] == package}
        folder = output / "source_sample" / package
        folder.mkdir(parents=True)
        allowed_ids = set()
        for name in ("video_nodes", "aspect_nodes", "evidence_nodes"):
            records = [
                clean_value(row) for row in tables[name]
                if row["video_id"].removeprefix("video:") in chosen
            ]
            write_csv(folder / (name + ".csv"), records, list(tables[name][0]))
            id_field = {"video_nodes": "video_id", "aspect_nodes": "aspect_id",
                        "evidence_nodes": "evidence_id"}[name]
            allowed_ids.update(row[id_field] for row in records)
        allowed_ids.update(chosen)
        allowed_ids.update("video:" + identity for identity in chosen)
        records = [
            clean_value(row) for row in tables["candidate_edges"]
            if (row.get("source_node_id") or row.get("source_id")) in allowed_ids
            and (row.get("target_node_id") or row.get("target_id")) in allowed_ids
        ]
        write_csv(folder / "candidate_edges.csv", records, list(tables["candidate_edges"][0]))
    write_json(output / "audit.json", report)
    hashes = {
        str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in output.rglob("*") if path.is_file()
    }
    write_json(output / "manifest.json", {
        "schema_version": 2, "outputs": hashes,
        "inputs": {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(input_files)
        },
        "scripts": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(__file__).parent.glob("*.py")
        },
    })
    print(json.dumps({
        "output": str(output), "scope": report["scope"],
        "videos": len(rows), "split_counts": report["split_counts"],
        "excluded_support_edges": len(changes), "training_ready": False,
    }, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Explicitly opt into the full frozen cohort.")
    parser.add_argument("--rules", type=Path, default=Path(__file__).with_name("review_rules.json"))
    parser.add_argument("--output", type=Path, help="New run directory under artifacts\\isolated_revision.")
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, KeyError, OSError, csv.Error, ArithmeticError) as error:
        print(f"Isolation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
