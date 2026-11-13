"""Regression guards for the concrete report/packaging audit findings."""
import json,tempfile,unittest,sys
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'Scripts')]
from isolate_revision import campaign_groups,proposed_split,read_csv,load_json
from experiment_results import available_experiments
from package_project import collect_paths


class AuditRegressionTests(unittest.TestCase):
    def test_maxims_and_all_registered_families_are_closed_components(self):
        rows=read_csv(ROOT/'data/processed/videos_v2.csv',[])
        splits={r['youtube_id']:r for r in read_csv(ROOT/'artifacts/v2/splits.csv',[])}
        rules=load_json(ROOT/'Scripts/review_rules.json')
        groups,pending,_=campaign_groups(rows,splits,rules);assign=proposed_split(groups,pending)
        self.assertEqual(groups['CxbnjaET7ew'],groups['WBMGLbAvnQY'])
        for family in rules['campaign_families']:
            self.assertEqual(len({groups[key] for key in family['youtube_ids']}),1)
            self.assertEqual(len({assign[key] for key in family['youtube_ids']}),1)
            if family['status']=='needs_review':self.assertTrue(set(family['youtube_ids'])<=pending)

    def test_nonobject_optional_manifest_is_rejected_without_breaking_legacy(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);legacy=root/'artifacts/v2';legacy.mkdir(parents=True)
            for n in ['primary_gnn.pt','web_results.json','optimization.json','metrics.json']:(legacy/n).write_text('fixture')
            manifest=root/'artifacts/isolated_revision/bad/evaluation/workspace/artifacts/v2/app_manifest.json'
            manifest.parent.mkdir(parents=True);manifest.write_text('[]')
            errors=[];available=available_experiments(root,errors=errors)
            self.assertEqual(len(available),1);self.assertEqual(len(errors),1)
            self.assertIn('JSON object',errors[0])

    def test_real_package_inventory_includes_new_runtime_dependency(self):
        paths=collect_paths(ROOT)
        self.assertIn(ROOT/'Scripts/text_policy.py',paths)
        self.assertIn(ROOT/'AUTHORS.json',paths)
        config=json.loads((ROOT/'artifacts/final_v4.json').read_text())
        self.assertIn(ROOT/config['showcase_path'],paths)
        self.assertFalse(any('.local-tools' in p.parts for p in paths))


if __name__=='__main__':unittest.main()
