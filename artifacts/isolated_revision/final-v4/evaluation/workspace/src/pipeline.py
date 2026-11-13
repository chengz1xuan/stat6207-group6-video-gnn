from text_policy import response_free_text
"""Reproducible grouped, inductive advertising-view experiment. No response features."""
from pathlib import Path
import os,json,re,csv,copy,itertools,hashlib
os.environ.setdefault('MPLCONFIGDIR',str(Path('tmp/matplotlib').resolve()))
import numpy as np,pandas as pd,torch
from torch import nn
from sklearn.model_selection import GroupShuffleSplit,GroupKFold
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT);torch.set_num_threads(1)
CAT=['brand_family','creative_mechanic','structure_type','evidence_mode','cta_explicitness','proof_strength','pacing','production_complexity']
TEXT=['main_message','hook','value_promise','cta']
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda o: o.item() if isinstance(o,np.generic) else o.tolist()),encoding='utf8')
def read(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def normid(s):return s.removeprefix('video:')
def prepare():
 rows=[]; labels=[];excluded=[]; counts=[]
 folders=sorted(Path('data_topic3').glob('*/video_nodes.csv'))
 for p in folders:
  if 'hsbc_16_' in str(p) or 'hang_seng_20_clean_v0_1' in str(p):continue
  stats={}
  for e in read(p.parent/'evidence_nodes.csv'):
   if e.get('kind')=='engagement_stats':
    try:s=json.loads(e.get('value_json') or e['text'])
    except (ValueError,KeyError):continue
    stats[normid(e['video_id'])]=(s,e)
  raw=read(p);counts.append({'package':p.parent.name,'rows':len(raw)})
  for r in raw:
   vid=normid(r['video_id']); s,e=stats.get(vid,({},{}));v=s.get('views')
   if v is None and r.get('views'):v=float(r['views'])
   if v is None:excluded.append({'video_id':vid,'reason':'missing_views'});continue
   z={**r,'video_id':vid,'brand_family':r.get('brand_family') or r.get('brand_canonical'),'creative_mechanic':r.get('creative_mechanic') or r.get('creative_mechanic_label'),'hook':r.get('hook') or r.get('hook_expression',''),'cta':r.get('cta') or r.get('cta_expression',''),'product_campaign':r.get('product_campaign') or r.get('product_or_campaign',''),'views':int(v),'captured_at':e.get('captured_at',''),'label_evidence':e.get('evidence_id',''),'package':p.parent.name}
   labels.append({k:z.get(k,'') for k in ['video_id','source_url','views','captured_at','label_evidence','package']});rows.append(z)
 df=pd.DataFrame(rows).fillna('');df['youtube_id']=df.source_url.str.extract(r'(?:v=|youtu.be/)([\w-]+)',expand=False).fillna(df.video_id)
 dup=df[df.duplicated('youtube_id',keep='last')];df=df.drop_duplicates('youtube_id',keep='last').sort_values('youtube_id').reset_index(drop=True)
 # Conservative campaign grouping, always within brand. URL dedup happens first.
 df['group']=df.brand_family+'|'+df.product_campaign.map(lambda x:re.sub(r'[^\w]+',' ',x.lower()).strip() if x else '')
 df.loc[df.product_campaign=='','group']=df.brand_family+'|'+df.video_id
 # Union exact descriptive duplicates as well as campaign aliases.
 parent=list(range(len(df)))
 def find(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 for key in ['group','main_message']:
  seen={}
  for i,r in df.iterrows():
   val=(r.brand_family,str(r[key]).strip().lower())
   if not val[1]:continue
   if val in seen:parent[find(i)]=find(seen[val])
   else:seen[val]=i
 df['group']=[str(find(i)) for i in range(len(df))]
 for c in CAT:df[c]=df[c].replace('','unknown')
 df['duration_seconds']=pd.to_numeric(df.duration_seconds,errors='coerce')
 pub=pd.to_datetime(df.publish_date,utc=True,errors='coerce'); cap=pd.to_datetime(df.captured_at,utc=True,errors='coerce')
 df['age_days']=(cap-pub).dt.total_seconds()/86400;df.loc[df.age_days<0,'age_days']=np.nan
 df.to_csv('data/processed/videos.csv',index=False);pd.DataFrame(labels).to_csv('data/processed/label_provenance.csv',index=False)
 audit={'packages':counts,'raw_retained_rows':len(rows),'unique_videos':len(df),'brands':df.brand_family.nunique(),'groups':df.group.nunique(),'duplicate_urls_removed':len(dup),'excluded':excluded,'missing_age':int(df.age_days.isna().sum()),'snapshot_min':str(cap.min()),'snapshot_max':str(cap.max()),'views_min':int(df.views.min()),'views_max':int(df.views.max()),'note':'Archived source snapshots; not a common-time recrawl. Latest HSBC/Hang Seng packages only.'};dump('artifacts/audit.json',audit)
 return df,audit
class Features:
 def fit(self,d):
  self.enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False).fit(d[CAT]);n=self.numeric(d);self.med=np.nanmedian(n,axis=0);self.scaler=StandardScaler().fit(np.where(np.isnan(n),self.med,n));self.vec=TfidfVectorizer(max_features=800,ngram_range=(1,2),min_df=2,sublinear_tf=True);t=self.vec.fit_transform(self.text(d));self.svd=TruncatedSVD(n_components=min(16,t.shape[1]-1),random_state=42).fit(t);return self
 def text(self, d):
  return [' '.join(response_free_text(r.get(c, '')) for c in TEXT) for _, r in d.iterrows()]
 def numeric(self,d):return np.column_stack([np.log1p(pd.to_numeric(d.duration_seconds,errors='coerce').clip(lower=0)),np.log1p(pd.to_numeric(d.age_days,errors='coerce').clip(lower=0))])
 def transform(self,d):
  n=self.numeric(d);missing=np.isnan(n).astype(float);n=np.where(np.isnan(n),self.med,n)
  return np.column_stack([self.enc.transform(d[CAT]),self.scaler.transform(n),missing,self.svd.transform(self.vec.transform(self.text(d)))]).astype('float32')
 def metadata(self,d):
  x=self.transform(d);b=len(self.enc.categories_[0]);off=sum(map(len,self.enc.categories_));return np.column_stack([x[:,:b],x[:,off:off+4]])
def neighbors(q,r,k=5,self_ref=False):
 q=q/(np.linalg.norm(q,axis=1,keepdims=True)+1e-9);r=r/(np.linalg.norm(r,axis=1,keepdims=True)+1e-9);s=q@r.T
 if self_ref:np.fill_diagonal(s,-np.inf)
 return np.argsort(-s,axis=1,kind='stable')[:,:min(k,len(r)-int(self_ref))]
class Net(nn.Module):
 def __init__(self,p,graph=True,h=24):
  super().__init__();self.graph=graph;self.l1=nn.Linear(p*(2 if graph else 1),h);self.l2=nn.Linear(h*(2 if graph else 1),1)
 def hidden(self,x,idx):return torch.relu(self.l1(torch.cat([x,x[idx].mean(1)],1) if self.graph else x))
 def forward(self,x,idx,q=None,qi=None):
  h=self.hidden(x,idx)
  if q is None:return self.l2(torch.cat([h,h[idx].mean(1)],1) if self.graph else h).squeeze(1)
  qh=torch.relu(self.l1(torch.cat([q,x[qi].mean(1)],1) if self.graph else q))
  return self.l2(torch.cat([qh,h[qi].mean(1)],1) if self.graph else qh).squeeze(1)
def train(x,y,v,vy,graph,seed=42,wd=.01):
 torch.manual_seed(seed);m=Net(x.shape[1],graph);opt=torch.optim.Adam(m.parameters(),lr=.01,weight_decay=wd);xt=torch.tensor(x);vt=torch.tensor(v);idx=torch.tensor(neighbors(x,x,self_ref=True));vi=torch.tensor(neighbors(v,x));ym=float(y.mean());ys=float(y.std());yt=torch.tensor((y-ym)/ys,dtype=torch.float32);best=1e99;wait=0;history=[]
 for epoch in range(600):
  m.train();opt.zero_grad();pred=m(xt,idx);loss=((pred-yt)**2).mean();loss.backward();opt.step();m.eval()
  with torch.no_grad():vp=m(xt,idx,vt,vi).numpy()*ys+ym
  val=float(np.mean((vp-vy)**2));history.append([epoch,float(loss.detach()),val])
  if val<best-1e-5:best=val;state=copy.deepcopy(m.state_dict());wait=0;bestepoch=epoch
  else:wait+=1
  if wait>=60:break
 m.load_state_dict(state)
 def predict(z):
  with torch.no_grad():return m(xt,idx,torch.tensor(z),torch.tensor(neighbors(z,x))).numpy()*ys+ym
 return m,predict,{'best_epoch':bestepoch,'validation_log_rmse':best**.5,'history':history,'y_mean':ym,'y_std':ys,'seed':seed,'weight_decay':wd}
def metric(y,p):
 p=np.clip(p,0,25);a=np.expm1(y);b=np.expm1(p);rho=float(spearmanr(y,p).statistic) if len(y)>1 and np.std(p)>1e-8 and np.std(y)>1e-8 else None
 return {'log_MAE':float(mean_absolute_error(y,p)),'log_RMSE':float(mean_squared_error(y,p)**.5),'Spearman':rho,'views_MAE':float(mean_absolute_error(a,b)),'views_RMSE':float(mean_squared_error(a,b)**.5),'log_R2':float(r2_score(y,p)) if len(y)>1 else None}
def main():
 df,audit=prepare();y=np.log1p(df.views.to_numpy(float));allidx=np.arange(len(df));dev,te=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(df,groups=df.group));a,b=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=43).split(df.iloc[dev],groups=df.iloc[dev].group));tr,va=dev[a],dev[b]
 for a,b in [(tr,va),(tr,te),(va,te)]:assert not set(df.iloc[a].group)&set(df.iloc[b].group)
 df['split']='train';df.loc[va,'split']='validation';df.loc[te,'split']='test';df[['video_id','youtube_id','group','split']].to_csv('artifacts/splits.csv',index=False)
 # Three campaign-grouped folds inside training only, for hyperparameter tuning.
 cv={name:{} for name in ['Ridge','Metadata Ridge','MLP','GraphSAGE']}
 for it,iv in GroupKFold(n_splits=3).split(df.iloc[tr],groups=df.iloc[tr].group):
  ti,vi=tr[it],tr[iv];ff=Features().fit(df.iloc[ti]);xx=ff.transform(df)
  for name,meta in [('Ridge',False),('Metadata Ridge',True)]:
   zz=ff.metadata(df) if meta else xx
   for alpha in [.1,1,10,100]:
    mm=Ridge(alpha=alpha).fit(zz[ti],y[ti]);score=float(mean_squared_error(y[vi],mm.predict(zz[vi]))**.5);cv[name].setdefault(str(alpha),[]).append(score)
  for name,g in [('MLP',False),('GraphSAGE',True)]:
   for wd in [.001,.01,.1]:
    _,_,cc=train(xx[ti],y[ti],xx[vi],y[vi],g,42,wd);cv[name].setdefault(str(wd),[]).append(cc['validation_log_rmse'])
 chosen={n:float(min(scores,key=lambda k:np.mean(scores[k]))) for n,scores in cv.items()};dump('artifacts/cross_validation.json',{'folds':3,'scope':'training partition only; fold-specific transforms and neighbors','scores':cv,'selected':chosen})
 f=Features().fit(df.iloc[tr]);x=f.transform(df);xt,xv,xe=x[tr],x[va],x[te];ym=y[tr].mean();preds={'Mean':np.full(len(te),ym)};validation={'Mean':metric(y[va],np.full(len(va),ym))};configs={};models={}
 for name,meta in [('Ridge',False),('Metadata Ridge',True)]:
  xx=f.metadata(df) if meta else x;best=None
  for alpha in [chosen[name]]:
   m=Ridge(alpha=alpha).fit(xx[tr],y[tr]);score=mean_squared_error(y[va],m.predict(xx[va]));
   if best is None or score<best[0]:best=(score,m,alpha)
  preds[name]=best[1].predict(xx[te]);validation[name]=metric(y[va],best[1].predict(xx[va]));configs[name]={'alpha':best[2]}
 for name,g in [('MLP',False),('GraphSAGE',True)]:
  runs=[]
  for wd in [chosen[name]]:
   m,p,c=train(xt,y[tr],xv,y[va],g,42,wd);runs.append((c['validation_log_rmse'],m,p,c))
  _,m,p,c=min(runs,key=lambda z:z[0]);models[name]=(m,p,c);preds[name]=p(xe);validation[name]=metric(y[va],p(xv));configs[name]={k:v for k,v in c.items() if k!='history'}
 results={n:metric(y[te],p) for n,p in preds.items()};dump('artifacts/metrics.json',{'test':results,'validation':validation,'configs':configs,'split_sizes':{s:int(sum(df.split==s)) for s in ['train','validation','test']},'selected_by_validation':min(validation,key=lambda n:validation[n]['log_RMSE'])})
 m,p,c=models['GraphSAGE'];torch.save({'state_dict':m.state_dict(),'features':f,'reference_features':xt,'reference_ids':df.iloc[tr].video_id.tolist(),'config':c,'categories':CAT},'artifacts/graphsage.pt')
 out=df.iloc[te][['video_id','title','brand_family','source_url','views']].copy()
 for n,pr in preds.items():out[n+'_log']=pr;out[n+'_views']=np.expm1(np.clip(pr,0,25))
 out['absolute_log_error']=abs(y[te]-preds['GraphSAGE']);out.sort_values('absolute_log_error',ascending=False).to_csv('artifacts/test_predictions.csv',index=False)
 # Paired bootstrap, conditional on this split/model; not population coverage.
 rng=np.random.default_rng(44);boot=[];delta=[]
 for _ in range(2000):
  ix=rng.integers(0,len(te),len(te));boot.append(np.sqrt(np.mean((y[te][ix]-preds['GraphSAGE'][ix])**2)));delta.append(np.mean(abs(y[te][ix]-preds['GraphSAGE'][ix]))-np.mean(abs(y[te][ix]-preds['Ridge'][ix])))
 uncertainty={'graph_log_RMSE_95_percentile':np.percentile(boot,[2.5,97.5]).tolist(),'paired_MAE_graph_minus_ridge_95':np.percentile(delta,[2.5,97.5]).tolist(),'note':'Video-level bootstrap conditional on fixed holdout; correlated campaigns and split uncertainty are not covered.'}
 seedresults=[]
 for seed in [7,21,84]:
  _,pp,cc=train(xt,y[tr],xv,y[va],True,seed,c['weight_decay']);seedresults.append({'seed':seed,**metric(y[te],pp(xe))})
 uncertainty['additional_initialization_runs']=seedresults;dump('artifacts/uncertainty.json',uncertainty)
 errors=out.assign(log_views=y[te],pred_log=preds['GraphSAGE'])
 slices=[]
 for col,series in [('brand',errors.brand_family),('view_scale',pd.cut(errors.views,[0,1e4,1e5,1e6,np.inf],include_lowest=True).astype(str))]:
  for name in series.unique():
   mask=series==name;sl=errors.loc[mask];slices.append({'type':col,'slice':name,'n':len(sl),**metric(sl.log_views.values,sl.pred_log.values)})
 dump('artifacts/error_slices.json',slices)
 # Search only observed supported triples, with at most two categorical edits.
 fields=['creative_mechanic','structure_type','pacing'];combos=df.iloc[tr].groupby(fields).size();combos=[tuple(k) for k,v in combos.items() if v>=2]
 def optimize(row):
  candidates=[row.copy()];changes=[{}]
  for comb in combos:
   ch={k:v for k,v in zip(fields,comb) if v!=row[k]}
   if not 0<len(ch)<=2:continue
   z=row.copy()
   for k,v in ch.items():z[k]=v
   candidates.append(z);changes.append(ch)
  cs=pd.DataFrame(candidates);ps=p(f.transform(cs));j=int(np.argmax(ps));return {'video_id':row.video_id,'title':row.title,'brand':row.brand_family,'split':row.split,'observed_views':int(row.views),'base_prediction':float(np.expm1(np.clip(ps[0],0,25))),'optimized_prediction':float(np.expm1(np.clip(ps[j],0,25))),'changes':changes[j],'candidates_evaluated':len(cs),'support_examples':df.iloc[tr].loc[(df.iloc[tr][fields]==cs.iloc[j][fields]).all(axis=1),['video_id','title','source_url']].to_dict('records')[:3]}
 cases=[optimize(df.iloc[i]) for i in te[:6]];new=df.iloc[tr[0]].copy();new['video_id']='new-video-example';new['title']='New video scenario (training reference template)';new['split']='new';new['views']=0;new['duration_seconds']=30;new['age_days']=30;newcase=optimize(new);newcase.pop('observed_views');dump('artifacts/optimization.json',{'new_video':newcase,'dataset_cases':cases,'restriction':'At most two edits among creative mechanic, structure and pacing; candidate triples occur >=2 times in training. Text/context fixed; counterfactual consistency not guaranteed. No causal or realized uplift claim.'})
 # Browser model export: actual weights, train references, preprocessing, and support.
 export={'categories':{k:v.tolist() for k,v in zip(CAT,f.enc.categories_)},'cat_fields':CAT,'text_fields':TEXT,'median':f.med.tolist(),'mean':f.scaler.mean_.tolist(),'scale':f.scaler.scale_.tolist(),'vocabulary':f.vec.vocabulary_,'idf':f.vec.idf_.tolist(),'components':f.svd.components_.tolist(),'reference_x':xt.tolist(),'reference_ids':df.iloc[tr].video_id.tolist(),'weights':{k:v.detach().numpy().tolist() for k,v in m.state_dict().items()},'y_mean':c['y_mean'],'y_std':c['y_std'],'k':5,'supported_combos':[list(z) for z in combos]}
 dump('website/dist/model.json',export)
 # Two graph views: canonical concept membership and actual model neighbor edges.
 concepts={};links=[]
 for i,r in df.iterrows():
  for field in fields+['cta_explicitness','evidence_mode']:
   key=field+':'+r[field];concepts[key]={'id':key,'field':field,'value':r[field]};links.append({'video':r.video_id,'concept':key})
 videos=[]
 for i,r in df.iterrows():
  z={k:(None if pd.isna(r[k]) else r[k]) for k in CAT+TEXT+['video_id','title','source_url','duration_seconds','age_days','views','split','product_campaign','captured_at','label_evidence','package']};videos.append(z)
 edge=neighbors(xt,xt,self_ref=True);model_edges=[{'source':df.iloc[tr[i]].video_id,'target':df.iloc[tr[j]].video_id} for i in range(len(tr)) for j in edge[i]]
 dump('website/dist/results.json',{'audit':audit,'metrics':json.loads(Path('artifacts/metrics.json').read_text()),'uncertainty':uncertainty,'videos':videos,'concepts':list(concepts.values()),'membership_edges':links,'model_edges':model_edges,'cases':cases,'new_video':newcase,'slices':slices})
 dump('artifacts/graph_summary.json',{'video_nodes':len(df),'concept_nodes':len(concepts),'membership_edges':len(links),'training_directed_neighbor_edges':len(model_edges),'concept_fields':fields+['cta_explicitness','evidence_mode']})
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,ax=plt.subplots(figsize=(7,4));names=list(results);ax.barh(names,[results[n]['log_RMSE'] for n in names],color=['#b4c5c5']*4+['#167c80']);ax.set_xlabel('Held-out RMSE on log(1 + views), lower is better');fig.tight_layout();fig.savefig('artifacts/model_comparison.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(6,4));ax.scatter(y[te],preds['GraphSAGE'],c='#167c80',alpha=.8);lims=[min(y[te].min(),preds['GraphSAGE'].min())-.4,max(y[te].max(),preds['GraphSAGE'].max())+.4];ax.plot(lims,lims,'--',c='#888');ax.set(xlabel='Observed log(1 + views)',ylabel='GraphSAGE prediction',xlim=lims,ylim=lims);fig.tight_layout();fig.savefig('artifacts/prediction_scatter.png',dpi=180);plt.close(fig)
 dump('artifacts/run_manifest.json',{'python':__import__('sys').executable,'torch':torch.__version__,'numpy':np.__version__,'seed_split':42,'seed_validation':43,'feature_fit_split':'train only','source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('data_topic3').glob('*/*.csv'))},'pipeline_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
 print(json.dumps({'audit':audit,'test_metrics':results,'sizes':{s:int(sum(df.split==s)) for s in df.split.unique()}},indent=2))
if __name__=='__main__':main()
