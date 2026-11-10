from pathlib import Path
import sys,json,os
import numpy as np,pandas as pd,torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pipeline import Features,Net,neighbors
root=Path(__file__).resolve().parents[1];os.chdir(root)
c=torch.load('artifacts/graphsage.pt',weights_only=False,map_location='cpu');m=Net(c['reference_features'].shape[1]);m.load_state_dict(c['state_dict']);m.eval();f=c['features'];x=c['reference_features'];idx=neighbors(x,x,self_ref=True);r=json.loads(Path('website/dist/results.json').read_text());d=pd.DataFrame(r['videos']);split=pd.read_csv('artifacts/splits.csv');
for a,b in [('train','test'),('train','validation'),('validation','test')]:
 assert not set(split.loc[split.split==a,'group'])&set(split.loc[split.split==b,'group'])
assert not split.youtube_id.duplicated().any();assert len(d)==219;assert (d.views>=0).all()
assert set(c['reference_ids'])==set(split.loc[split.split=='train','video_id'])
q=f.transform(d)
with torch.no_grad():p=m(torch.tensor(x),torch.tensor(idx),torch.tensor(q),torch.tensor(neighbors(q,x))).numpy()*c['config']['y_std']+c['config']['y_mean']
fixture=[{'row':row,'features':q[i].tolist(),'expected_log':float(p[i])} for i,row in enumerate(r['videos'])]
Path('artifacts/inference_fixture.json').write_text(json.dumps(fixture,ensure_ascii=False))
pred=pd.read_csv('artifacts/test_predictions.csv');lookup=dict(zip(d.video_id,p));assert max(abs(lookup[v]-z) for v,z in zip(pred.video_id,pred.GraphSAGE_log))<1e-4
print('PASS: group / URL disjointness, label counts, reference isolation, checkpoint-test-prediction agreement.')
