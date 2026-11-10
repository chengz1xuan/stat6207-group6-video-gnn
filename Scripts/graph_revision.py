"""Rebuild source-backed graphs with an explicit conservative evidence policy."""

from collections import Counter

from text_policy import response_free_text


ALIASES = {
    "hook_expression": "hook", "cta_expression": "cta", "proof_type": "proof",
    "audience_market_signal": "audience_signal", "brand_node_candidate": "brand_family",
    "brand": "brand_family", "product_or_campaign": "product_campaign",
}
ASPECTS = {
    "brand_family", "creative_mechanic", "structure_type", "evidence_mode",
    "cta_explicitness", "proof_strength", "pacing", "production_complexity",
    "main_message", "hook", "value_promise", "cta", "proof", "visual_signal",
    "audience_signal",
}
# Summaries/takeaways are annotations, not independent observations of the claim.
OBSERVATIONS = {
    "description_text", "extract_spoken_dialogue", "extract_on_screen_text",
    "extract_end_card_text",
}
EDGE_TYPES = {
    "video_has_aspect": ("video", "aspect"),
    "video_has_evidence": ("video", "evidence"),
    "aspect_supported_by_evidence": ("aspect", "evidence"),
}
VIDEO_RELATIONS = {
    "same_brand", "same_canonical_brand", "same_author", "same_evidence_mode",
    "same_creative_mechanic", "same_cta_explicitness", "same_pacing",
    "same_structure_type", "same_product_or_campaign",
}


def rebuild_graph(row, tables, rules, issues):
    identity, vid = row["youtube_id"], row["video_id"]
    roots = [r for r in tables["video_nodes"] if r["video_id"].removeprefix("video:") == vid]
    if len(roots) != 1:
        raise ValueError(f"Expected one original video root: {identity}")
    root = roots[0]["video_id"]
    nodes = [{"id": root, "type": "video", "field": "video", "text": ""}]
    canonical = {root: root, vid: root, "video:" + vid: root}
    seen, counts = {}, Counter()
    summary = Counter()
    for kind, name in (("aspect", "aspect_nodes"), ("evidence", "evidence_nodes")):
        for record in tables[name]:
            if record["video_id"].removeprefix("video:") != vid:
                continue
            field = (
                ALIASES.get(record.get("aspect_type"), record.get("aspect_type"))
                if kind == "aspect" else record.get("kind")
            )
            if field not in (ASPECTS if kind == "aspect" else OBSERVATIONS):
                summary["excluded_non_observation_or_field"] += 1
                continue
            text = response_free_text(record.get("aspect_value") or record.get("text") or "")
            if kind == "aspect" and field == "hook" and row["hook_status"] == "duplicate_summary":
                summary["duplicate_hook_omitted"] += 1
                continue
            if kind == "aspect" and field == "proof" and (
                text == response_free_text(row["value_promise"])
            ):
                summary["value_promise_not_independent_proof"] += 1
                continue
            if not text:
                summary["empty_or_response_text"] += 1
                continue
            node_id = record["aspect_id"] if kind == "aspect" else record["evidence_id"]
            signature = (kind, field, text)
            if signature in seen:
                canonical[node_id] = seen[signature]
                summary["duplicate_nodes_remapped"] += 1
                continue
            limit = 3 if kind == "evidence" else 4 if field in {"visual_signal", "audience_signal"} else 1
            if counts[(kind, field)] >= limit:
                summary["fixed_cap"] += 1
                continue
            counts[(kind, field)] += 1
            canonical[node_id] = node_id
            seen[signature] = node_id
            nodes.append({"id": node_id, "type": kind, "field": field, "text": text})
    lookup = {node["id"]: node for node in nodes}
    excluded = {
        rule["source_edge_id"]: rule for rule in rules["excluded_support_edges"]
        if rule["youtube_id"] == identity
    }
    found = set()
    changes, edges = [], []
    for source in tables["candidate_edges"]:
        a = source.get("source_node_id") or source.get("source_id")
        b = source.get("target_node_id") or source.get("target_id")
        edge_type = source["edge_type"]
        if edge_type not in EDGE_TYPES:
            continue
        # Review explicit exclusions even when their claim node is also removed.
        if source["edge_id"] in excluded:
            found.add(source["edge_id"])
            changes.append({
                "youtube_id": identity, "source_edge_id": source["edge_id"],
                "action": "exclude_reviewed_link", "reason": excluded[source["edge_id"]]["reason"],
            })
            continue
        if a not in canonical or b not in canonical:
            continue
        ca, cb = canonical[a], canonical[b]
        if (lookup[ca]["type"], lookup[cb]["type"]) != EDGE_TYPES[edge_type]:
            raise ValueError(f"Invalid source edge endpoint types: {source['edge_id']}")
        heuristic = edge_type == "aspect_supported_by_evidence" and (
            source.get("deterministic", "").lower() != "true"
        )
        if heuristic:
            changes.append({
                "youtube_id": identity, "source_edge_id": source["edge_id"],
                "action": "quarantine_heuristic_support",
                "reason": "Not independently reviewed; retained in the review ledger, not the model graph.",
            })
            continue
        edges.append({
            "source": ca, "target": cb, "type": edge_type, "heuristic": False,
            "source_edge_id": source["edge_id"],
            "original_source": a, "original_target": b,
            "endpoint_remapped": (ca, cb) != (a, b),
        })
    if found != set(excluded):
        raise ValueError(f"Stale reviewed edge IDs for {identity}: {set(excluded) - found}")
    unique = {}
    for edge in edges:
        key = (edge["source"], edge["target"], edge["type"])
        if key in unique:
            unique[key].setdefault("additional_source_edges", []).append({
                field: edge[field] for field in (
                    "source_edge_id", "original_source", "original_target", "endpoint_remapped",
                )
            })
        else:
            unique[key] = edge
    edges = list(unique.values())
    connected = {endpoint for edge in edges for endpoint in (edge["source"], edge["target"])}
    for node in nodes[1:]:
        if node["id"] not in connected:
            summary["unconnected_source_node_omitted"] += 1
    nodes = [node for node in nodes if node["id"] == root or node["id"] in connected]
    if not edges:
        raise ValueError(f"No usable source-backed graph edges: {identity}")
    if not any(edge["type"] == "aspect_supported_by_evidence" for edge in edges):
        issues.append({
            "kind": "no_concept_evidence_support", "youtube_id": identity,
            "reason": "No surviving direct source support; membership graph retained without invented links.",
        })
    issues.append({"kind": "graph_policy_counts", "youtube_id": identity, **summary})
    return {
        "video_id": vid, "package": row["package"], "nodes": nodes, "edges": edges,
    }, changes


def video_pairs(source_tables, eligible_ids):
    pairs = set()
    for tables in source_tables.values():
        roots = {r["video_id"].removeprefix("video:") for r in tables["video_nodes"]}
        for edge in tables["candidate_edges"]:
            if edge["edge_type"] not in VIDEO_RELATIONS:
                continue
            a = (edge.get("source_node_id") or edge.get("source_id")).removeprefix("video:")
            b = (edge.get("target_node_id") or edge.get("target_id")).removeprefix("video:")
            if a != b and a in roots and b in roots and {a, b} <= eligible_ids:
                pairs.add(tuple(sorted((a, b))))
    return sorted(pairs)
