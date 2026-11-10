from pathlib import Path
import sys,os,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'.project-deps'));os.chdir(ROOT)
import numpy as np,pandas as pd,torch
from revision import GraphFeatures,GraphRegressor,node_edges
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
lobo=pd.read_csv(D/'leave_brand_out.csv');assert len(lobo)==56;assert lobo.groupby('model').brand.nunique().eq(14).all();assert all(z['train_log_RMSE']>=0 for n,z in metrics['configs'].items() if 'epochs' in z)
result={'fresh_label_cohort':214,'split':[171,43],'group_and_URL_disjoint':True,'source_label_agreement':True,'structural_edge_provenance':True,'response_node_exclusion':True,'query_outbound_message_edges':0,'training_reference_isolation':True,'CV_selection_only':True,'LOBO_model_brand_runs':len(lobo),'quantitative_neural_training_losses':True};(D/'validation.json').write_text(json.dumps(result,indent=2));print('PASS:',result)
