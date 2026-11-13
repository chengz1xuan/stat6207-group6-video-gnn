"""Validate final report inputs, grouped evaluation, graph provenance and showcase parity."""
from pathlib import Path
import sys,json,hashlib,csv
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'.project-deps'),str(R/'Scripts')]
import numpy as np,pandas as pd
from model_service import ModelService
from experiment_results import validate_bundle
from final_experiment import configuration,showcase,data_dictionary,report_matches
from run_paired_experiment import validate_run


def main():
    load=lambda p:json.loads(p.read_text());config=configuration();D=R/config['primary_folder'];prepared=R/config['prepared_directory'];P=R/config['paired_directory']
    validate_bundle(D);s=ModelService(D);m=s.results['metrics'];raw=pd.read_csv(prepared/'videos.csv');folds=pd.read_csv(D/'inner_folds.csv')
    train=set(raw.loc[raw.split=='train','video_id']);test=set(raw.loc[raw.split=='test','video_id']);pending=set(raw.loc[raw.split=='quarantine','video_id'])
    assert (len(train),len(test),len(pending))==(161,40,13)
    assert not train&test and not train&pending and not test&pending
    assert train==s.train_ids and train|test==set(s.videos)
    for _,part in raw.groupby('group'):assert part.split.nunique()==1
    ids=raw.set_index('youtube_id');rules=load(prepared/'review_rules.json')
    for family in rules['campaign_families']:
        part=ids.loc[family['youtube_ids']];assert part.group.nunique()==1 and part.split.nunique()==1
        if family['status']=='needs_review':assert set(part.video_id)<=pending
    for _,part in folds.groupby('fold'):
        fit=part[part.role=='fit'];val=part[part.role=='validation'];assert set(part.video_id)==train
        assert not set(fit.group)&set(val.group)
    fresh=pd.read_csv(R/'data/youtube/snapshot.csv').set_index('video_id').views
    for row in s.videos.values():assert row['views']==fresh[row['video_id']]
    assert m['selected_primary_gnn']==min(['Subgraph structural','Subgraph + similarity'],key=m['CV_log_RMSE'].get)
    predictions=pd.read_csv(D/'test_predictions.csv');assert set(predictions.video_id)==test
    y=np.log1p(predictions.views.to_numpy(float))
    for name,z in m['test'].items():
        p=predictions[name+'_log'].to_numpy();assert abs(np.mean((p-y)**2)**.5-z['log_RMSE'])<1e-5
    measurements=pd.read_csv(D/'training_measurements.csv');assert len(measurements)==10 and measurements.train_log_RMSE.notna().all()
    pairs=load(prepared/'video_pairs.json');audit=s.results['brand_edge_audit'];assert len(pairs)==audit['core_undirected_pairs']==1382
    assert all(s.videos[a]['brand_family']==s.videos[b]['brand_family'] for a,b in pairs)
    assert audit['no_brand_pairs']==0 and audit['equals_complete_brand_cliques']
    source_ids={}
    for g in s.graphs.values():
        if g['package'] not in source_ids:
            with (R/'data_topic3'/g['package']/'candidate_edges.csv').open(encoding='utf-8-sig') as f:source_ids[g['package']]={r['edge_id'] for r in csv.DictReader(f)}
        assert all(e['source_edge_id'] in source_ids[g['package']] for e in g['edges'])
        assert all(not e['heuristic'] for e in g['edges'])
    diagnostics=load(D/'report_diagnostics.json');display=diagnostics['display_graph'];graph=s.graphs[display['video_id']]
    assert display['video_id'] in train
    assert all(n in graph['nodes'] for n in display['nodes']) and all(e in graph['edges'] for e in display['source_edges'])
    assert any(e['type']=='aspect_supported_by_evidence' for e in display['source_edges'])
    _,added=s.features.graph(pd.Series(s.videos[display['video_id']]),graph,True)
    expected={tuple(sorted((graph['nodes'][a]['id'],graph['nodes'][b]['id']))) for a,b in added}
    assert all(tuple(pair) in expected for pair in display['semantic_edges'])
    validate_run(P);summary=load(P/'summary.json');assert summary['mode']=='full_retrained_paired' and summary['outputs']
    for relative,expected_hash in summary['outputs'].items():assert hashlib.sha256((P/relative).read_bytes()).hexdigest()==expected_hash
    assert load(P/'comparison.json')['shared_control_baselines_match']
    left=pd.read_csv(P/'v2/workspace/artifacts/v2/test_predictions.csv').set_index('video_id')
    right=pd.read_csv(P/'v3/workspace/artifacts/v2/test_predictions.csv').set_index('video_id')
    assert set(left.index)==set(right.index)==test
    for name in ['Mean','Brand mean','Metadata Ridge']:assert np.max(abs(left[name+'_log']-right[name+'_log']))<1e-6
    assert [c['video_id'] for c in s.results['cases']]==predictions.video_id.tolist()[:6]
    for case in s.results['cases']:
        row=s.videos[case['video_id']];result=s.optimize(row,s.graphs[row['video_id']])
        assert abs(result['best']['log']-case['optimized_log'])<1e-4
        assert all(e['video_id'] in train for e in case['support_examples'])
    brief=showcase(D);result=s.optimize(brief['row'],s.query_graph(brief['row']))
    assert abs(result['base']['log']-brief['base_log'])<1e-4 and abs(result['best']['log']-brief['optimized_log'])<1e-4
    assert brief['observed_views'] is None and all(n['type']!='evidence' for n in s.query_graph(brief['row'])['nodes'])
    assert brief['row']['hook'] and len(brief['changes'])<=2
    assert report_matches(D) and not report_matches(R/'artifacts/v2')
    assert data_dictionary(D)==D/'DATA_DICTIONARY.md'
    assert len(pd.read_csv(D/'leave_brand_out.csv'))==70
    # PDF validation is separate from numerical/model validation if the optional renderer is absent.
    import pymupdf
    pdf=pymupdf.open(R/config['report_path']);assert len(pdf)==4
    text='\n'.join(page.get_text() for page in pdf)
    assert all(abs(p.rect.width-595.28)<.1 and abs(p.rect.height-841.89)<.1 for p in pdf)
    for author in load(R/'AUTHORS.json'):assert author.get('ieee_name',author['name']) in text and author['email'] in text
    assert 'Group 6' not in text and 'GROUP 6' not in text
    for value in ['2.678','2.650','161','40','13','316,494','580,104']:assert value in text
    result={'version':'v4','cohort':201,'split':[161,40],'quarantine':13,'registered_families':len(rules['campaign_families']),
        'outer_and_inner_family_isolation':True,'fresh_label_agreement':True,'same_split_ten_model_measurements':True,
        'paired_controls_and_hashes':True,'brand_pairs':len(pairs),'remaining_no_brand_pairs':0,'real_graph_source_and_semantic_edges':True,
        'six_case_and_declared_brief_parity':True,'LOBO_runs':70,'named_authors_and_emails':4,'report_pages':len(pdf),'A4':True}
    (D/'v4_validation.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS',result)


if __name__=='__main__':main()
