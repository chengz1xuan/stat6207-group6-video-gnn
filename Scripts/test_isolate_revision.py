import copy
import tempfile
import unittest
from pathlib import Path

from isolate_revision import (
    ROOT, campaign_groups, check_label, clean_value, exact_count, keyed,
    proposed_split, read_csv, safe_output, youtube_id,
)
from graph_revision import rebuild_graph


class IsolationTests(unittest.TestCase):
    def test_exact_counts(self):
        self.assertEqual(exact_count("123.0"), 123)
        for value in ("123.5", "-1", "NaN", "Infinity"):
            with self.assertRaises(ValueError):
                exact_count(value)

    def test_url_identity(self):
        self.assertEqual(youtube_id("https://youtu.be/PhVN9idahMQ"), "PhVN9idahMQ")
        for url in (
            "https://youtube.com.evil.example/watch?v=PhVN9idahMQ",
            "https://www.youtube.com/watch?v=PhVN9idahMQ&v=0_1xI0ihZSk",
            "https://www.youtube.com/watch?v=short",
        ):
            with self.assertRaises(ValueError):
                youtube_id(url)

    def test_duplicate_identity_rejected(self):
        with self.assertRaises(ValueError):
            keyed([{"id": "a"}, {"id": "a"}], "id")

    def test_clean_nested_nul(self):
        self.assertEqual(clean_value({"a": ["x\x00y"]}), {"a": ["xy"]})

    def test_csv_cleaning_does_not_modify_source(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "Scripts") as directory:
            path = Path(directory) / "source.csv"
            original = b"id,text\n1,a\x00b\n"
            path.write_bytes(original)
            issues = []
            self.assertEqual(read_csv(path, issues)[0]["text"], "ab")
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(issues[0]["count"], 1)

    def test_output_isolation_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            allowed = root / "artifacts" / "isolated_revision"
            self.assertEqual(safe_output(root, allowed / "new"), (allowed / "new").resolve())
            for path in (root, root / "data", allowed, allowed / ".." / "v2"):
                with self.assertRaises(ValueError):
                    safe_output(root, path)
            existing = allowed / "old"
            existing.mkdir(parents=True)
            with self.assertRaises(ValueError):
                safe_output(root, existing)

    def test_conservative_alias_union_and_transitive_quarantine(self):
        rows = [
            {"youtube_id": "a", "group": "1", "brand_family": "B",
             "product_campaign": "Product A", "main_message": "First"},
            {"youtube_id": "b", "group": "2", "brand_family": "B",
             "product_campaign": "Product B", "main_message": "Second"},
            {"youtube_id": "c", "group": "2", "brand_family": "B",
             "product_campaign": "Product C", "main_message": "Third"},
        ]
        splits = {row["youtube_id"]: {"group": row["group"], "split": "train"} for row in rows}
        rules = {"campaign_families": [{
            "campaign_id": "series", "status": "needs_review",
            "youtube_ids": ["a", "b"], "reason": "Pending",
        }]}
        groups, quarantine, _ = campaign_groups(rows, splits, rules)
        self.assertEqual(len(set(groups.values())), 1)
        self.assertEqual(quarantine, {"a", "b", "c"})

    def test_original_groups_never_split(self):
        rows = [
            {"youtube_id": key, "group": "1", "brand_family": "B",
             "product_campaign": key, "main_message": key}
            for key in ("a", "b")
        ]
        splits = {key: {"group": "1", "split": "train"} for key in ("a", "b")}
        groups, _, _ = campaign_groups(rows, splits, {"campaign_families": []})
        self.assertEqual(groups["a"], groups["b"])

    def test_nearest_whole_group_split_is_deterministic(self):
        groups = {str(i): f"group-{i // 3}" for i in range(12)}
        first = proposed_split(groups, set())
        self.assertEqual(first, proposed_split(dict(reversed(list(groups.items()))), set()))
        self.assertEqual(sum(value == "test" for value in first.values()), 3)
        for group in set(groups.values()):
            self.assertEqual(len({first[key] for key in groups if groups[key] == group}), 1)

    def test_quarantine_is_excluded_from_split_target(self):
        groups = {str(i): str(i) for i in range(10)}
        split = proposed_split(groups, {"8", "9"})
        self.assertEqual(sum(value == "test" for value in split.values()), 2)
        self.assertEqual(split["8"], "quarantine")
        with self.assertRaises(ValueError):
            proposed_split({"a": "same", "b": "same"}, set())

    def fixture_graph(self):
        row = {
            "video_id": "v", "youtube_id": "identity", "hook_status": "source_annotation",
            "value_promise": "Benefit", "package": "fixture",
        }
        source = {
            "video_nodes": [{"video_id": "v"}],
            "aspect_nodes": [
                {"video_id": "v", "aspect_id": "a", "aspect_type": "proof", "aspect_value": "Claim"},
            ],
            "evidence_nodes": [
                {"video_id": "v", "evidence_id": "e", "kind": "extract_on_screen_text",
                 "text": "Unrelated"},
            ],
            "candidate_edges": [
                {"source_id": "v", "target_id": "a", "edge_type": "video_has_aspect", "edge_id": "1"},
                {"source_id": "v", "target_id": "e", "edge_type": "video_has_evidence", "edge_id": "2"},
                {"source_id": "a", "target_id": "e", "edge_type": "aspect_supported_by_evidence",
                 "edge_id": "3", "deterministic": "false"},
            ],
        }
        rules = {"excluded_support_edges": [{
            "youtube_id": "identity", "source_edge_id": "3", "reason": "Not supported",
        }]}
        return row, source, rules

    def test_reviewed_edge_removal_preserves_original(self):
        row, source, rules = self.fixture_graph()
        original = copy.deepcopy(source)
        issues = []
        revised, changes = rebuild_graph(row, source, rules, issues)
        self.assertEqual(len(revised["edges"]), 2)
        self.assertEqual(len(changes), 1)
        self.assertEqual(source, original)
        self.assertEqual(issues[0]["kind"], "no_concept_evidence_support")

    def test_stale_review_and_bad_provenance_fail(self):
        row, source, rules = self.fixture_graph()
        rules["excluded_support_edges"][0]["source_edge_id"] = "missing"
        with self.assertRaises(ValueError):
            rebuild_graph(row, source, rules, [])
        rules["excluded_support_edges"] = []
        source["candidate_edges"][0]["target_id"] = "e"
        with self.assertRaises(ValueError):
            rebuild_graph(row, source, rules, [])

    def test_response_nodes_excluded(self):
        row, source, rules = self.fixture_graph()
        source["evidence_nodes"].append({
            "video_id": "v", "evidence_id": "response", "kind": "engagement_stats", "text": "100 views",
        })
        graph, _ = rebuild_graph(row, source, rules, [])
        self.assertNotIn("response", {node["id"] for node in graph["nodes"]})

    def test_unknown_support_quarantined_but_observation_retained(self):
        row, source, rules = self.fixture_graph()
        rules["excluded_support_edges"] = []
        graph, changes = rebuild_graph(row, source, rules, [])
        self.assertEqual(len(graph["nodes"]), 3)
        self.assertEqual(changes[0]["action"], "quarantine_heuristic_support")
        self.assertEqual(len(graph["edges"]), 2)

    def test_deterministic_observation_support_retained(self):
        row, source, rules = self.fixture_graph()
        rules["excluded_support_edges"] = []
        source["candidate_edges"][2]["deterministic"] = "true"
        graph, _ = rebuild_graph(row, source, rules, [])
        self.assertEqual(len(graph["edges"]), 3)

    def test_derived_evidence_not_treated_as_observation(self):
        row, source, rules = self.fixture_graph()
        rules["excluded_support_edges"] = []
        source["evidence_nodes"][0]["kind"] = "extract_audience_takeaway"
        graph, _ = rebuild_graph(row, source, rules, [])
        self.assertNotIn("e", {node["id"] for node in graph["nodes"]})

    def test_duplicate_summary_hook_and_value_promise_proof_removed(self):
        row, source, rules = self.fixture_graph()
        rules["excluded_support_edges"] = []
        row["value_promise"] = "Claim"
        row["hook_status"] = "duplicate_summary"
        source["aspect_nodes"].append({
            "video_id": "v", "aspect_id": "hook", "aspect_type": "hook", "aspect_value": "Claim",
        })
        graph, _ = rebuild_graph(row, source, rules, [])
        self.assertEqual({node["type"] for node in graph["nodes"]}, {"video", "evidence"})

    def test_label_mismatch_fails(self):
        row = {
            "video_id": "v", "youtube_id": "PhVN9idahMQ",
            "source_url": "https://www.youtube.com/watch?v=PhVN9idahMQ",
            "views": "100", "captured_at": "2026-11-09T12:00:00+00:00",
            "retrieved_at_utc": "2026-11-09T12:00:00+00:00",
        }
        snapshot = dict(row)
        evidence = {**row, "evidence": {"views": {"originalViewCount": "100"}}}
        check_label(row, snapshot, evidence, row)
        evidence["views"] = "101"
        with self.assertRaises(ValueError):
            check_label(row, snapshot, evidence, row)


if __name__ == "__main__":
    unittest.main()
