from pathlib import Path
import sys,json
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'.project-deps'))
import numpy as np,pandas as pd,torch
from revision import GraphFeatures,GraphRegressor,pooled
from pipeline import Features
from sklearn.linear_model import Ridge
D=R/'artifacts/v2';cp=torch.load(D/'primary_gnn.pt',weights_only=False,map_location='cpu');f=cp['features'];d=json.loads((D/'web_results.json').read_text());df=pd.DataFrame(d['videos']);ids=set(cp['train_ids']);ix=[i for i,r in enumerate(d['videos']) if r['video_id'] in ids];train=df.iloc[ix];gs=[d['graphs'][i] for i in ix];y=np.log1p(train.views.to_numpy(float));m=d['metrics'];rows=[]
for name,meas in m['test'].items():
 c=m['configs'].get(name,{})
 if name=='Mean':p=np.full(len(y),y.mean());rmse=float(np.mean((p-y)**2)**.5);objective='MSE on ln(1+views); closed-form constant'
 elif 'Ridge' in name:
  x=f.root.metadata(train) if name=='Metadata Ridge' else pooled(f.transform(train,gs));ridge=Ridge(alpha=c['alpha']).fit(x,y);rmse=float(np.mean((ridge.predict(x)-y)**2)**.5);objective='sum squared log residuals + alpha * squared coefficient norm'
 else:rmse=c['train_log_RMSE'];objective='MSE on train-standardized ln(1+views)'
 rows.append({'model':name,'target':'ln(1+views)','objective':objective,'n_train':len(train),'train_log_RMSE':rmse,'CV_log_RMSE':m['CV_log_RMSE'].get(name),'test_log_RMSE':meas['log_RMSE'],'refit_epochs':c.get('epochs'),'regularization':c.get('alpha',c.get('weight_decay',0))})
pd.DataFrame(rows).to_csv(D/'training_measurements.csv',index=False);print('Saved quantitative training targets and measurements for all eight final models.')
