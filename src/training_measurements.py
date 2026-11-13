"""Export objectives and measured training/CV/test losses for a chosen frozen experiment."""
from pathlib import Path
import sys,json,argparse
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'.project-deps'))
import numpy as np,pandas as pd
from model_service import ModelService
from revision import pooled,brand_mean_predictions
from sklearn.linear_model import Ridge


def export(folder):
    service=ModelService(folder);f=service.features;d=service.results
    train=pd.DataFrame([v for k,v in service.videos.items() if k in service.train_ids])
    gs=[service.graphs[k] for k in train.video_id];y=np.log1p(train.views.to_numpy(float));m=d['metrics'];rows=[]
    for name,meas in m['test'].items():
        c=m['configs'].get(name,{})
        if name=='Mean':p=np.full(len(y),y.mean());rmse=float(np.mean((p-y)**2)**.5);objective='MSE on ln(1+views); closed-form constant'
        elif name=='Brand mean':p=brand_mean_predictions(train,train,y);rmse=float(np.mean((p-y)**2)**.5);objective='MSE on ln(1+views); training brand mean with global unseen-brand fallback'
        elif 'Ridge' in name:
            x=f.root.metadata(train) if name=='Metadata Ridge' else pooled(f.transform(train,gs));ridge=Ridge(alpha=c['alpha']).fit(x,y);rmse=float(np.mean((ridge.predict(x)-y)**2)**.5);objective='sum squared log residuals + alpha * squared coefficient norm'
        else:rmse=c['train_log_RMSE'];objective='MSE on train-standardized ln(1+views)'
        rows.append({'model':name,'target':'ln(1+views)','objective':objective,'n_train':len(train),'train_log_RMSE':rmse,'CV_log_RMSE':m['CV_log_RMSE'].get(name),'test_log_RMSE':meas['log_RMSE'],'refit_epochs':c.get('epochs'),'regularization':c.get('alpha',c.get('weight_decay',0))})
    pd.DataFrame(rows).to_csv(Path(folder)/'training_measurements.csv',index=False)
    print('Saved quantitative training targets and measurements for',len(rows),'models.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--folder',type=Path,default=R/'artifacts/v2');args=parser.parse_args();export(args.folder)
