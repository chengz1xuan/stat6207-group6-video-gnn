from pathlib import Path
import sys,os,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'.project-deps'));os.chdir(ROOT)
import numpy as np,pandas as pd,torch
from revision import GraphFeatures,GraphRegressor,node_edges,brand_mean_predictions,remove_brand_pairs,apply_campaign_overrides
from pipeline import Features
D=Path('artifacts/v2');splits=pd.read_csv(D/'splits.csv');df=pd.read_csv('data/processed/videos_v2.csv');raw=json.loads((D/'raw_graphs.json').read_text());snapshot=pd.read_csv('data/youtube/snapshot.csv');metrics=json.loads((D/'metrics.json').read_text());cv=json.loads((D/'cross_validation.json').read_text())
tr=splits[splits.split=='train'];te=splits[splits.split=='test'];assert (len(tr),len(te))==(171,43);assert not set(tr.group)&set(te.group);assert not set(tr.youtube_id)&set(te.youtube_id);assert splits.video_id.nunique()==214
fresh=snapshot.dropna(subset=['views']).set_index('video_id').views;assert all(fresh[r.video_id]==r.views for _,r in df.iterrows());assert len(fresh)==len(df)
for g in raw:
 ids={n['id'] for n in g['nodes']};assert len(ids)==len(g['nodes']);assert all(e['source'] in ids and e['target'] in ids and e['source_edge_id'] for e in g['edges']);assert all(n['field'] not in ['engagement_stats','fit_score','views','likes','comments','safety_status','copy_risk'] for n in g['nodes'])
audit=json.loads((D/'graph_audit.json').read_text());assert sum(audit['relations'].values())==audit['provided_edges'];assert sum(len(g['nodes']) for g in raw)==audit['nodes']
# Inference edge isolation: query indices can receive messages but never send to train/queries.
ids=tr.video_id.tolist();qs=te.video_id.tolist();x=np.random.default_rng(1).normal(size=(len(ids),5));q=np.random.default_rng(2).normal(size=(len(qs),5));e=node_edges(ids,qs,set(),x,q,True);assert all(int(s)<len(ids) for s in e[0])
cp=torch.load(D/'primary_gnn.pt',weights_only=False,map_location='cpu');assert set(cp['train_ids'])==set(tr.video_id);assert set(cp['test_ids'])==set(te.video_id);assert min(metrics['CV_log_RMSE'],key=metrics['CV_log_RMSE'].get)==metrics['selected_model']
pr=pd.read_csv(D/'test_predictions.csv');assert set(pr.video_id)==set(te.video_id);train_mean=float(np.log1p(df.set_index('video_id').loc[tr.video_id,'views']).mean());assert np.max(abs(pr['Mean_log']-train_mean))<1e-8
lobo=pd.read_csv(D/'leave_brand_out.csv');assert len(lobo)==70;assert lobo.groupby('model').brand.nunique().eq(14).all();assert all(z['train_log_RMSE']>=0 for n,z in metrics['configs'].items() if 'epochs' in z)
# Regression guards for the specific review findings.
overrides=pd.read_csv('data/processed/campaign_overrides.csv')
joined=splits.set_index('video_id').loc[overrides.video_id]
assert joined.group.nunique()==1 and joined.split.nunique()==1
folds=pd.read_csv(D/'inner_folds.csv')
for fold,part in folds.groupby('fold'):
 fit=part[part.role=='fit'];val=part[part.role=='validation']
 assert not set(fit.group)&set(val.group)
 assert set(part.video_id)==set(tr.video_id)
 for _,aliases in overrides.groupby('canonical_campaign_id'):
  selected=part[part.video_id.isin(aliases.video_id)]
  assert selected.role.nunique()<=1
byid=df.set_index('video_id');expected=brand_mean_predictions(byid.loc[tr.video_id],byid.loc[pr.video_id],np.log1p(byid.loc[tr.video_id].views.to_numpy(float)))
assert np.max(abs(expected-pr['Brand mean_log']))<1e-8
unseen=byid.loc[pr.video_id].copy();unseen['brand_family']='UNSEEN_BRAND'
assert np.max(abs(brand_mean_predictions(byid.loc[tr.video_id],unseen,np.log1p(byid.loc[tr.video_id].views.to_numpy(float)))-train_mean))<1e-8
edgeaudit=json.loads((D/'brand_edge_audit.json').read_text());edges=pd.read_csv(D/'video_edge_provenance.csv')
pairs={tuple(sorted((r.source,r.target))) for _,r in edges.iterrows()}
assert len(pairs)==edgeaudit['core_undirected_pairs']==1558
assert len(remove_brand_pairs(df,pairs))==edgeaudit['no_brand_pairs']==0
assert edgeaudit['equals_complete_brand_cliques']
ages=json.loads((D/'age_support.json').read_text());trainages=byid.loc[tr.video_id].age_days
assert ages['training_min_days']==trainages.min() and ages['training_median_days']==trainages.median()
assert ages['training_min_days']>30
history=json.loads((D/'reupload_error_audit.json').read_text())
assert history['observed_views']==1019 and abs(history['pre_correction_predicted_views']-442536.47)<1
assert history['corrected_split']=='train' and history['corrected_prediction_status'].startswith('In-sample')
reportgraph=json.loads((D/'report_graph.json').read_text());source=next(g for g in raw if g['video_id']==reportgraph['video_id'])
assert reportgraph['video_id'] in set(tr.video_id)
assert all(n in source['nodes'] for n in reportgraph['nodes']) and all(e in source['edges'] for e in reportgraph['edges'])
assert all(e['source'] in {n['id'] for n in reportgraph['nodes']} and e['target'] in {n['id'] for n in reportgraph['nodes']} for e in reportgraph['edges'])
result={'manual_bilingual_alias_outer_and_inner_safe':True,'training_only_brand_mean_and_unseen_fallback':True,'brand_clique_provenance_and_zero_edge_ablation':True,'training_age_support_and_30_day_OOD':True,'historical_reupload_separated_from_corrected_holdout':True,'report_graph_uses_original_nodes_edges':True,'fresh_label_cohort':214,'split':[171,43],'group_and_URL_disjoint':True,'source_label_agreement':True,'structural_edge_provenance':True,'response_node_exclusion':True,'query_outbound_message_edges':0,'training_reference_isolation':True,'CV_selection_only':True,'LOBO_model_brand_runs':len(lobo),'quantitative_neural_training_losses':True};(D/'validation.json').write_text(json.dumps(result,indent=2));print('PASS:',result)
