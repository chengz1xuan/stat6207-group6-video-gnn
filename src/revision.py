"""Topic 3 revision: public snapshot; exact grouped 80/20; graph-level PyG baseline."""
from pathlib import Path
import sys,os,json,re,copy,csv,hashlib,collections,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'.project-deps'));os.chdir(ROOT)
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/matplotlib'))
import numpy as np,pandas as pd,torch
from torch import nn
from torch_geometric.data import Data,Batch
from torch_geometric.nn import SAGEConv,global_mean_pool
from sklearn.model_selection import GroupShuffleSplit,GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from pipeline import Features,CAT,TEXT,normid,read,metric,dump,neighbors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
torch.set_num_threads(1);OUT=Path('artifacts/v2');OUT.mkdir(exist_ok=True)
ASPECT_ALIASES={'hook_expression':'hook','cta_expression':'cta','proof_type':'proof','audience_market_signal':'audience_signal','brand_node_candidate':'brand_family','brand':'brand_family','product_or_campaign':'product_campaign'}
ALLOWED_ASPECT=set(CAT+TEXT+['proof','visual_signal','audience_signal'])
ALLOWED_EVIDENCE={'description_text','extract_main_message','extract_cta','extract_audience_takeaway','extract_spoken_dialogue','extract_on_screen_text','extract_end_card_text'}
STRUCTURAL={'video_has_aspect','video_has_evidence','aspect_supported_by_evidence'}
VIDEO_REL={'same_brand','same_canonical_brand','same_author','same_evidence_mode','same_creative_mechanic','same_cta_explicitness','same_pacing','same_structure_type','same_product_or_campaign'}
BAD=re.compile(r'\b(views?|likes?|comments?|shares?|viral|followers?|subscribers?)\b|播放|觀看|点赞|點讚',re.I)
def safe_text(v):
 s=str(v)
 try:
  v=json.loads(s)
  if isinstance(v,str):s=v
 except ValueError:pass
 return ' '.join(z for z in re.split(r'[.!?。！？\n]',s) if not BAD.search(z)).strip()
def load_data():
 df=pd.read_csv('data/processed/videos.csv');snap=pd.read_csv('data/youtube/snapshot.csv');summary=json.loads(Path('data/youtube/summary.json').read_text());df['archived_views']=df.views;df=df.merge(snap[['video_id','views','likes','comments','retrieved_at_utc','channel_id','channel_name','subscriber_count_display','platform_publish_date','status']],on='video_id',suffixes=('','_youtube'),how='left');df['views']=df.views_youtube
 excluded=df[df.views.isna()][['video_id','source_url','status_youtube']].fillna('').to_dict('records');df=df[df.views.notna()].reset_index(drop=True);df['views']=df.views.astype(int);df['captured_at']=df.retrieved_at_utc;df['publish_date_archived']=df.publish_date;df['publish_date']=df.platform_publish_date.fillna(df.publish_date)
 pub=pd.to_datetime(df.publish_date,utc=True,errors='coerce',format='mixed');cap=pd.to_datetime(df.captured_at,utc=True,errors='coerce',format='mixed');df['age_days']=(cap-pub).dt.total_seconds()/86400;df.loc[df.age_days<0,'age_days']=np.nan
 for k in CAT+TEXT+['title','product_campaign']:df[k]=df[k].fillna('unknown' if k in CAT else '')
 df.to_csv('data/processed/videos_v2.csv',index=False)
 audit={'inventory':len(snap),'fresh_label_cohort':len(df),'brands':df.brand_family.nunique(),'groups':df.group.nunique(),'excluded_missing_fresh_views':excluded,'snapshot':summary,'archived_view_labels':219,'archived_likes':None,'archived_comments':None,'missing_age':int(df.age_days.isna().sum())};likes=comments=0
 for package in df.package.unique():
  for r in read(Path('data_topic3')/package/'evidence_nodes.csv'):
   if r.get('kind')=='engagement_stats':
    z=json.loads(r.get('value_json') or r['text']);likes+=z.get('likes') is not None;comments+=z.get('comments') is not None
 audit['archived_likes']=likes;audit['archived_comments']=comments;dump(OUT/'audit.json',audit);return df,audit

def raw_graphs(df):
 graphs={};video_edges=set();relation_counts=collections.Counter();filter_counts=collections.Counter();edge_audit=[]
 for package in df.package.unique():
  folder=Path('data_topic3')/package;vr=read(folder/'video_nodes.csv');ar=read(folder/'aspect_nodes.csv');er=read(folder/'evidence_nodes.csv');cr=read(folder/'candidate_edges.csv');roots={normid(r['video_id']):r['video_id'] for r in vr};allids={}
  for vid,rawid in roots.items():
   graphs[vid]={'video_id':vid,'nodes':[{'id':rawid,'type':'video','field':'video','text':''}],'edges':[],'package':package};allids[rawid]=vid;allids['video:'+vid]=vid;allids[vid]=vid
  seen=collections.defaultdict(set);counts=collections.Counter()
  for kind,records in [('aspect',ar),('evidence',er)]:
   for r in records:
    vid=normid(r['video_id']);field=ASPECT_ALIASES.get(r.get('aspect_type'),r.get('aspect_type')) if kind=='aspect' else r.get('kind');allowed=field in (ALLOWED_ASPECT if kind=='aspect' else ALLOWED_EVIDENCE)
    if not allowed:filter_counts['excluded_field']+=1;continue
    value=safe_text(r.get('aspect_value') or r.get('text') or r.get('label',''));limit=4 if field in ['visual_signal','audience_signal'] else (3 if kind=='evidence' else 1)
    if not value or counts[(vid,kind,field)]>=limit or (kind,field,value) in seen[vid]:filter_counts['empty_duplicate_or_cap']+=1;continue
    seen[vid].add((kind,field,value));counts[(vid,kind,field)]+=1;nodeid=r['aspect_id'] if kind=='aspect' else r['evidence_id'];graphs[vid]['nodes'].append({'id':nodeid,'type':kind,'field':field,'text':value});allids[nodeid]=vid
  allowedids={vid:{r['id'] for r in g['nodes']} for vid,g in graphs.items() if g['package']==package}
  for r in cr:
   source=r.get('source_node_id') or r.get('source_id');target=r.get('target_node_id') or r.get('target_id');etype=r['edge_type'];s,t=allids.get(source),allids.get(target)
   if etype in VIDEO_REL and normid(source) in roots and normid(target) in roots:
    video_edges.add(tuple(sorted([normid(source),normid(target)])))
   if s is None or s!=t or etype not in STRUCTURAL:continue
   # Allow root aliases only, retaining original provenance.
   ss=roots[s] if normid(source)==s else source;tt=roots[t] if normid(target)==t else target
   if ss not in allowedids[s] or tt not in allowedids[s]:continue
   heuristic=(etype=='aspect_supported_by_evidence' and r.get('deterministic','').lower()!='true')
   graphs[s]['edges'].append({'source':ss,'target':tt,'type':etype,'heuristic':heuristic,'source_edge_id':r['edge_id']});relation_counts[etype]+=1
   edge_audit.append({'video_id':s,'source':ss,'target':tt,'type':etype,'source_package':package,'source_edge_id':r['edge_id'],'heuristic':heuristic})
 gs=[graphs[r.video_id] for _,r in df.iterrows()]
 for g in gs:
  ids={r['id'] for r in g['nodes']};assert all(e['source'] in ids and e['target'] in ids for e in g['edges'])
 edge_audit=[r for r in edge_audit if r['video_id'] in set(df.video_id)];relation_counts=collections.Counter(e['type'] for g in gs for e in g['edges'])
 pd.DataFrame(edge_audit).to_csv(OUT/'edge_provenance.csv',index=False);dump(OUT/'graph_audit.json',{'graphs':len(gs),'nodes':sum(len(g['nodes']) for g in gs),'provided_edges':sum(len(g['edges']) for g in gs),'relations':dict(relation_counts),'excluded_nodes':dict(filter_counts),'node_video_core_pairs':len(video_edges),'policy':'Allowlisted response-free creative concepts/evidence; caps and duplicate rules fixed before tuning. Original membership/support edges only. Support links may be heuristic. Exclude engagement_stats, assessment scores, risk annotations, cross-video heuristic suggestions from primary per-video graphs.'});dump(OUT/'raw_graphs.json',gs);return gs,video_edges
class GraphFeatures:
 def fit(self,df,graphs):
  self.root=Features().fit(df);self.fields=sorted({n['field'] for g in graphs for n in g['nodes']});texts=[n['text'] for g in graphs for n in g['nodes'] if n['type']!='video'];self.vec=TfidfVectorizer(max_features=1000,ngram_range=(1,2),min_df=2,sublinear_tf=True).fit(texts);t=self.vec.transform(texts);self.svd=TruncatedSVD(n_components=16,random_state=42).fit(t);self.rdim=self.root.transform(df.iloc[:1]).shape[1];return self
 def graph(self,row,g,sim=False):
  n=len(g['nodes']);x=np.zeros((n,self.rdim+3+len(self.fields)+16),dtype='float32');x[0,:self.rdim]=self.root.transform(pd.DataFrame([row]))[0];tx=self.svd.transform(self.vec.transform([n['text'] for n in g['nodes']]));x[:,-16:]=tx
  for i,node in enumerate(g['nodes']):
   x[i,self.rdim+['video','aspect','evidence'].index(node['type'])]=1
   if node['field'] in self.fields:x[i,self.rdim+3+self.fields.index(node['field'])]=1
  ids={n['id']:i for i,n in enumerate(g['nodes'])};pairs=set()
  for e in g['edges']:
   a,b=ids[e['source']],ids[e['target']];pairs.add((a,b));pairs.add((b,a))
  added=[]
  if sim:
   # Fixed within-video semantic extension: threshold .55, top 2 per nonroot node.
   z=tx/(np.linalg.norm(tx,axis=1,keepdims=True)+1e-9);cs=z@z.T
   for a in range(1,n):
    choices=[b for b in np.argsort(-cs[a],kind='stable') if b!=a and b>0 and cs[a,b]>=.55][:2]
    for b in choices:
     if (a,b) not in pairs:added.append((a,b))
     pairs.add((a,b));pairs.add((b,a))
  edge=np.array(sorted(pairs),dtype='int64').T if pairs else np.empty((2,0),dtype='int64');d=Data(x=torch.tensor(x),edge_index=torch.tensor(edge));d.root=torch.tensor([0]);return d,added
 def transform(self,df,graphs,sim=False):return [self.graph(row,g,sim)[0] for (_,row),g in zip(df.iterrows(),graphs)]
 def export(self):
  f=self.root
  return {'root':{'categories':{k:v.tolist() for k,v in zip(CAT,f.enc.categories_)},'cat_fields':CAT,'text_fields':TEXT,'median':f.med.tolist(),'mean':f.scaler.mean_.tolist(),'scale':f.scaler.scale_.tolist(),'vocabulary':f.vec.vocabulary_,'idf':f.vec.idf_.tolist(),'components':f.svd.components_.tolist()},'node':{'vocabulary':self.vec.vocabulary_,'idf':self.vec.idf_.tolist(),'components':self.svd.components_.tolist()},'fields':self.fields,'root_dim':self.rdim,'similarity_threshold':.55,'similarity_k':2}
def pooled(ds):return np.stack([np.r_[d.x[0].numpy(),d.x.mean(0).numpy()] for d in ds])
class GraphRegressor(nn.Module):
 def __init__(self,p,message=True):
  super().__init__();self.message=message
  if message:self.c1=SAGEConv(p,24);self.c2=SAGEConv(24,24)
  else:self.c1=nn.Linear(p,24);self.c2=nn.Linear(24,24)
  self.head=nn.Linear(48,1)
 def forward(self,b):
  h=torch.relu(self.c1(b.x,b.edge_index) if self.message else self.c1(b.x));h=torch.relu(self.c2(h,b.edge_index) if self.message else self.c2(h));return self.head(torch.cat([h[b.ptr[:-1]],global_mean_pool(h,b.batch)],1)).squeeze(1)
class NodeRegressor(nn.Module):
 def __init__(self,p):super().__init__();self.c1=SAGEConv(p,24);self.c2=SAGEConv(24,1)
 def forward(self,x,e):return self.c2(torch.relu(self.c1(x,e)),e).squeeze(1)
def edge_index(ids,pairs):
 lookup={v:i for i,v in enumerate(ids)};es=set()
 for a,b in pairs:
  if a in lookup and b in lookup and a!=b:es.add((lookup[a],lookup[b]));es.add((lookup[b],lookup[a]))
 return es

def node_edges(train_ids,query_ids,pairs,xt,xq,sim):
 es=edge_index(train_ids,pairs);lookup={v:i for i,v in enumerate(train_ids)};qlookup={v:len(train_ids)+i for i,v in enumerate(query_ids)}
 for a,b in pairs:
  if a in lookup and b in qlookup:es.add((lookup[a],qlookup[b]))
  if b in lookup and a in qlookup:es.add((lookup[b],qlookup[a]))
 if sim:
  for i,js in enumerate(neighbors(xt,xt,self_ref=True)):
   for j in js:es.add((int(j),i))
  for i,js in enumerate(neighbors(xq,xt)) if len(query_ids) else []:
   for j in js:es.add((int(j),len(train_ids)+i))
 return torch.tensor(np.array(sorted(es),dtype='int64').T if es else np.empty((2,0),dtype='int64'))
def fit_neural(kind,trdata,y,vdata=None,vy=None,wd=.1,epochs=None,seed=42):
 torch.manual_seed(seed);isnode=kind.startswith('Node');graph=kind!='Pooled MLP'
 if isnode:
  xt,et=trdata;model=NodeRegressor(xt.shape[1]);xt=torch.tensor(xt);q,eq=vdata if vdata is not None else (None,None)
  def tp():return model(xt,et)
  def vp():return model(torch.tensor(np.concatenate([xt.numpy(),q])),eq)[len(xt):]
 else:
  bt=Batch.from_data_list(trdata);bv=Batch.from_data_list(vdata) if vdata else None;model=GraphRegressor(bt.num_features,graph)
  def tp():return model(bt)
  def vp():return model(bv)
 ym=float(y.mean());ys=float(y.std());target=torch.tensor((y-ym)/ys,dtype=torch.float32);opt=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=wd);best=1e99;wait=0;history=[]
 for ep in range(epochs if epochs is not None else 260):
  model.train();opt.zero_grad();loss=((tp()-target)**2).mean();loss.backward();opt.step()
  if epochs is not None:history.append([ep+1,float(loss.detach()),None])
  if epochs is None:
   model.eval()
   with torch.no_grad():pr=vp().numpy()*ys+ym
   score=float(np.mean((pr-vy)**2));history.append([ep+1,float(loss.detach()),score])
   if score<best-1e-5:best=score;state=copy.deepcopy(model.state_dict());best_ep=ep+1;wait=0
   else:wait+=1
   if wait>=35:break
 if epochs is None:model.load_state_dict(state)
 else:best_ep=epochs
 model.eval()
 with torch.no_grad():train_rmse=float(torch.sqrt(((tp()-target)**2).mean()))*ys
 return model,{'train_log_RMSE':train_rmse,'training_target':'standardized ln(1+views) MSE','y_mean':ym,'y_std':ys,'epochs':best_ep,'weight_decay':wd,'history':history}
def predict(model,c,data,reference=None):
 with torch.no_grad():
  if isinstance(model,NodeRegressor):
   xt,xq,e=data;z=model(torch.tensor(np.concatenate([xt,xq])),e)[len(xt):]
  else:z=model(Batch.from_data_list(data))
 return z.numpy()*c['y_std']+c['y_mean']
NAMES=['Mean','Metadata Ridge','Pooled Ridge','Pooled MLP','Subgraph structural','Subgraph + similarity','Node structural','Node + similarity']
def build_fold(df,gs,train,val,pairs):
 f=GraphFeatures().fit(df.iloc[train],[gs[i] for i in train]);a=f.transform(df.iloc[train],[gs[i] for i in train]);b=f.transform(df.iloc[val],[gs[i] for i in val]);sa=f.transform(df.iloc[train],[gs[i] for i in train],True);sb=f.transform(df.iloc[val],[gs[i] for i in val],True);xt=f.root.transform(df.iloc[train]);xv=f.root.transform(df.iloc[val]);ids=df.iloc[train].video_id.tolist();qids=df.iloc[val].video_id.tolist();
 return f,a,b,sa,sb,xt,xv,ids,qids

def train_all(df,gs,tr,te,pairs):
 y=np.log1p(df.views.to_numpy(float));cv={n:{} for n in NAMES if n!='Mean'};oof={};folds=list(GroupKFold(3).split(tr,groups=df.iloc[tr].group))
 for fold,(aa,bb) in enumerate(folds):
  ti,vi=tr[aa],tr[bb];f,a,b,sa,sb,xt,xv,ids,qids=build_fold(df,gs,ti,vi,pairs)
  for name in NAMES[1:]:
   for param in ([10.,100.] if 'Ridge' in name else [.01,.1]):
    if 'Ridge' in name:
     tx,vx=(f.root.metadata(df.iloc[ti]),f.root.metadata(df.iloc[vi])) if name=='Metadata Ridge' else (pooled(a),pooled(b));model=Ridge(alpha=param).fit(tx,y[ti]);pr=model.predict(vx);config={'alpha':param}
    else:
     if name.startswith('Node'):
      sim='similarity' in name;et=node_edges(ids,[],pairs,xt,xt[:0],sim);eq=node_edges(ids,qids,pairs,xt,xv,sim);model,config=fit_neural(name,(xt,et),y[ti],(xv,eq),y[vi],wd=param);pr=predict(model,config,(xt,xv,eq))
     else:
      td,vd=(sa,sb) if name=='Subgraph + similarity' else (a,b);model,config=fit_neural(name,td,y[ti],vd,y[vi],wd=param);pr=predict(model,config,vd)
    cv[name].setdefault(str(param),[]).append({'fold':fold,'log_RMSE':metric(y[vi],pr)['log_RMSE'],'epochs':config.get('epochs'),'train_log_RMSE':config.get('train_log_RMSE')});oof.setdefault((name,param),np.full(len(df),np.nan))[vi]=pr
  print('CV fold',fold+1,'complete',flush=True)
 chosen={n:float(min(v,key=lambda p:np.mean([z['log_RMSE'] for z in v[p]]))) for n,v in cv.items()};dump(OUT/'cross_validation.json',{'folds':3,'scope':'outer 80% training only; fold-local transforms, graphs and stopping','scores':cv,'selected':chosen});f,a,b,sa,sb,xt,xv,ids,qids=build_fold(df,gs,tr,te,pairs);preds={'Mean':np.full(len(te),y[tr].mean())};models={};configs={};bands={}
 for name in NAMES[1:]:
  param=chosen[name]
  if 'Ridge' in name:
   tx,vx=(f.root.metadata(df.iloc[tr]),f.root.metadata(df.iloc[te])) if name=='Metadata Ridge' else (pooled(a),pooled(b));model=Ridge(alpha=param).fit(tx,y[tr]);pr=model.predict(vx);config={'alpha':param}
  else:
   epochs=int(np.median([z['epochs'] for z in cv[name][str(param)]]));
   if name.startswith('Node'):
    sim='similarity' in name;et=node_edges(ids,[],pairs,xt,xt[:0],sim);eq=node_edges(ids,qids,pairs,xt,xv,sim);model,config=fit_neural(name,(xt,et),y[tr],wd=param,epochs=epochs);pr=predict(model,config,(xt,xv,eq))
   else:
    td,vd=(sa,sb) if name=='Subgraph + similarity' else (a,b);model,config=fit_neural(name,td,y[tr],wd=param,epochs=epochs);pr=predict(model,config,vd)
  preds[name]=pr;models[name]=model;configs[name]=config;res=abs(oof[(name,param)][tr]-y[tr]);bands[name]=float(np.quantile(res,.9,method='higher'))
 results={n:metric(y[te],p) for n,p in preds.items()};cvmeans={n:np.mean([v['log_RMSE'] for v in cv[n][str(chosen[n])]]) for n in cv};selected=min(cvmeans,key=cvmeans.get);gnn=min(['Subgraph structural','Subgraph + similarity'],key=cvmeans.get);dump(OUT/'metrics.json',{'test':results,'CV_log_RMSE':cvmeans,'selected_model':selected,'selected_primary_gnn':gnn,'train_size':len(tr),'test_size':len(te),'configs':{n:{k:v for k,v in c.items() if k!='history'} for n,c in configs.items()},'OOF_90_absolute_log_residual':bands,'uncertainty_note':'OOF residual sensitivity bands, not guaranteed/calibrated individual prediction intervals.'});dump(OUT/'training_histories.json',{n:c.get('history',[]) for n,c in configs.items()})
 pd.DataFrame([{'video_id':df.iloc[i].video_id,'target_log':float(y[i]),**{n:float(oof[(n,chosen[n])][i]) for n in cv}} for i in tr]).to_csv(OUT/'oof_predictions.csv',index=False)
 out=df.iloc[te][['video_id','title','brand_family','source_url','views','group']].copy();
 for n,p in preds.items():out[n+'_log']=p;out[n+'_views']=np.expm1(np.clip(p,0,25))
 out.to_csv(OUT/'test_predictions.csv',index=False)
 torch.save({'state_dict':models[gnn].state_dict(),'features':f,'config':configs[gnn],'model_name':gnn,'train_ids':ids,'test_ids':qids,'graph_definition':'video subgraph with supplied structural relations; optional within-video semantic edges'},OUT/'primary_gnn.pt');
 # Group-resampled bootstrap, fixed trained models. Same sample for paired comparisons.
 rng=np.random.default_rng(44);groups=df.iloc[te].group.to_numpy();ug=np.unique(groups);ci={n:[] for n in NAMES};delta=[]
 for _ in range(2000):
  chosen_groups=rng.choice(ug,len(ug),replace=True);ix=np.concatenate([np.flatnonzero(groups==q) for q in chosen_groups]);
  for n in NAMES:ci[n].append(float(np.mean((preds[n][ix]-y[te][ix])**2)**.5))
  delta.append(float(np.mean(abs(preds['Subgraph + similarity'][ix]-y[te][ix]))-np.mean(abs(preds['Subgraph structural'][ix]-y[te][ix]))))
 unc={'test_campaign_groups':len(ug),'test_log_RMSE_95':{n:np.percentile(v,[2.5,97.5]).tolist() for n,v in ci.items()},'paired_log_MAE_similarity_minus_structural_95':np.percentile(delta,[2.5,97.5]).tolist(),'note':'Campaign-level percentile bootstrap, 2000 draws, conditional on this frozen snapshot and split; no split/model-selection uncertainty.'};dump(OUT/'uncertainty.json',unc)
 # Three fixed seeds; no selection based on the testing set.
 seeds=[]
 for seed in [7,21,84]:
  td,vd=(sa,sb) if gnn=='Subgraph + similarity' else (a,b);model,conf=fit_neural(gnn,td,y[tr],wd=configs[gnn]['weight_decay'],epochs=configs[gnn]['epochs'],seed=seed);seeds.append({'seed':seed,**metric(y[te],predict(model,conf,vd))})
 dump(OUT/'seed_sensitivity.json',seeds)
 return f,models,configs,preds,gnn,bands,results,unc

def leave_brand_out(df,gs,pairs):
 y=np.log1p(df.views.to_numpy(float));records=[];names=['Pooled Ridge','Pooled MLP','Subgraph structural','Subgraph + similarity']
 for brand in sorted(df.brand_family.unique()):
  tr=np.flatnonzero((df.brand_family!=brand).to_numpy());te=np.flatnonzero((df.brand_family==brand).to_numpy());aa,bb=next(GroupShuffleSplit(1,test_size=.2,random_state=73).split(tr,groups=df.iloc[tr].group));ti,vi=tr[aa],tr[bb];ff,a,b,sa,sb,*_=build_fold(df,gs,ti,vi,pairs);epochs={}
  for name in names[1:]:
   td,vd=(sa,sb) if name=='Subgraph + similarity' else (a,b);_,c=fit_neural(name,td,y[ti],vd,y[vi],wd=.1);epochs[name]=c['epochs']
  f,a,b,sa,sb,*_=build_fold(df,gs,tr,te,pairs)
  for name in names:
   if name=='Pooled Ridge':model=Ridge(alpha=100.).fit(pooled(a),y[tr]);p=model.predict(pooled(b));ep=None
   else:
    td,vd=(sa,sb) if name=='Subgraph + similarity' else (a,b);model,c=fit_neural(name,td,y[tr],wd=.1,epochs=epochs[name]);p=predict(model,c,vd);ep=epochs[name]
   records.append({'brand':brand,'model':name,'n_train':len(tr),'n_test':len(te),'epochs':ep,**metric(y[te],p)})
  print('LOBO',brand,'complete',flush=True)
 pd.DataFrame(records).to_csv(OUT/'leave_brand_out.csv',index=False);summary={n:{'brands':len(df.brand_family.unique()),'macro_log_MAE':float(np.mean([r['log_MAE'] for r in records if r['model']==n])),'macro_log_RMSE':float(np.mean([r['log_RMSE'] for r in records if r['model']==n])),'macro_Spearman':float(np.nanmean([r['Spearman'] if r['Spearman'] is not None else np.nan for r in records if r['model']==n]))} for n in names};dump(OUT/'leave_brand_out_summary.json',{'models':summary,'protocol':'Each brand held out entirely. Fixed alpha=100 / weight_decay=.1; inner group holdout chooses epochs, then refit other brands. No held-out brand labels used for preprocessing/stopping. Not tuned on main outer testing results.'});return summary

def edit_graph(g,row,changes):
 g=copy.deepcopy(g)
 for n in g['nodes']:
  if n['type']=='aspect' and n['field'] in changes:n['text']=str(row[n['field']])
 return g

def new_graph(row):
 root={'id':'new:video','type':'video','field':'video','text':''};nodes=[root];edges=[]
 for field in CAT+TEXT:
  nid='new:'+field;nodes.append({'id':nid,'type':'aspect','field':field,'text':safe_text(row[field])});edges.append({'source':root['id'],'target':nid,'type':'author_declared_membership','heuristic':False,'source_edge_id':None})
 return {'video_id':'new-video-example','package':'hypothetical new brief','nodes':nodes,'edges':edges}

def scenarios(df,gs,tr,te,f,model,config,gnn,band):
 fields=['creative_mechanic','structure_type','pacing'];counts=df.iloc[tr].groupby(fields).size();combos=[tuple(k) for k,v in counts.items() if v>=2];sim=gnn=='Subgraph + similarity'
 def optimize(row,g,observed=True):
  rows=[row.copy()];graphs=[g];changes=[{}]
  for combo in combos:
   ch={k:v for k,v in zip(fields,combo) if row[k]!=v}
   if not 0<len(ch)<=2:continue
   r=row.copy()
   for k,v in ch.items():r[k]=v
   rows.append(r);graphs.append(edit_graph(g,r,ch));changes.append(ch)
  data=[f.graph(r,gg,sim)[0] for r,gg in zip(rows,graphs)];p=predict(model,config,data);j=int(np.argmax(p));support=df.iloc[tr].loc[(df.iloc[tr][fields]==rows[j][fields]).all(axis=1),['video_id','title','source_url']].to_dict('records')[:3]
  z={'video_id':row.video_id,'title':row.title,'brand':row.brand_family,'brief_objective':'Increase observable video reach for the specified product while preserving brand, claims, duration and exposure horizon.','creative_elements':{k:row[k] for k in fields+['main_message','hook','cta']},'base_prediction':float(np.expm1(np.clip(p[0],0,25))),'optimized_prediction':float(np.expm1(np.clip(p[j],0,25))),'base_log':float(p[0]),'optimized_log':float(p[j]),'uncertainty_band':[float(np.expm1(max(0,p[0]-band))),float(np.expm1(min(25,p[0]+band)))],'changes':changes[j],'candidates_evaluated':len(rows),'support_examples':support,'transferable_takeaway':'Treat supported setting changes as a storyboard hypothesis; rewrite and re-annotate matching content before production, then compare variants under equal media exposure. No actual counterfactual outcome is observed.'}
  if observed:z['observed_views']=int(row.views)
  return z
 cases=[optimize(df.iloc[i],gs[i]) for i in te[:6]];r=df.iloc[tr[0]].copy();r['video_id']='new-video-example';r['title']='New advertising brief: training-reference product context';r['duration_seconds']=30;r['age_days']=30
 new=optimize(r,new_graph(r),False);new['storyboard']=[{'seconds':'0-3','purpose':'Hook','content':r['hook']},{'seconds':'3-20','purpose':'Demonstrate the product benefit','content':r['value_promise']},{'seconds':'20-27','purpose':'Apply suggested creative structure','content':'; '.join(k+' = '+v for k,v in new['changes'].items()) or 'Keep the current supported settings.'},{'seconds':'27-30','purpose':'CTA and end card','content':r['cta']}];dump(OUT/'optimization.json',{'new_video':new,'dataset_cases':cases,'supported_combos':combos,'rule':'At most two categorical changes among mechanic/structure/pacing. Triples occur >=2 times in outer training. Frozen GNN; recompute semantic edges. Evidence/text fixed for dataset cases; new graph uses only declared brief inputs. OOF 90% residual bands are sensitivity summaries, not guaranteed intervals.'});return cases,new,combos

def main():
 df,audit=load_data();gs,pairs=raw_graphs(df);n=len(df);ntest=round(n*.2)
 for seed in range(42,2042):
  tr,te=next(GroupShuffleSplit(1,test_size=.2,random_state=seed).split(df,groups=df.group))
  if len(te)==ntest:break
 else:raise ValueError('No group-safe 80/20 count found')
 assert not set(df.iloc[tr].group)&set(df.iloc[te].group);assert not set(df.iloc[tr].youtube_id)&set(df.iloc[te].youtube_id)
 df['split']='train';df.loc[te,'split']='test';df[['video_id','youtube_id','group','brand_family','split']].to_csv(OUT/'splits.csv',index=False);dump(OUT/'split_policy.json',{'train':len(tr),'test':len(te),'target_test_count':ntest,'seed':seed,'selection_rule':'First seed >=42 yielding round(0.2*N) test videos with disjoint campaign groups. Selection uses group sizes only, never response labels.','train_fraction':len(tr)/n,'test_fraction':len(te)/n})
 f,models,configs,preds,gnn,bands,results,unc=train_all(df,gs,tr,te,pairs);lobo=leave_brand_out(df,gs,pairs);cases,new,combos=scenarios(df,gs,tr,te,f,models[gnn],configs[gnn],gnn,bands[gnn]);
 ex={'model_name':gnn,'preprocessing':f.export(),'weights':{k:v.detach().numpy().tolist() for k,v in models[gnn].state_dict().items()},'y_mean':configs[gnn]['y_mean'],'y_std':configs[gnn]['y_std'],'residual_band':bands[gnn],'supported_combos':combos,'train_ids':df.iloc[tr].video_id.tolist()};dump(OUT/'web_model.json',ex)
 videos=[]
 for _,r in df.iterrows():
  z={k:None if pd.isna(r[k]) else r[k] for k in CAT+TEXT+['video_id','youtube_id','title','source_url','duration_seconds','age_days','views','archived_views','likes','comments','channel_id','channel_name','subscriber_count_display','captured_at','split','product_campaign']};videos.append(z)
 slices=[];y=np.log1p(df.views.to_numpy(float));pp=preds[gnn]
 for name,mask in [('views <= 10k',df.iloc[te].views.to_numpy()<=1e4),('10k < views <= 1m',(df.iloc[te].views.to_numpy()>1e4)&(df.iloc[te].views.to_numpy()<=1e6)),('views > 1m',df.iloc[te].views.to_numpy()>1e6)]:
  if mask.any():slices.append({'slice':name,'n':int(mask.sum()),**metric(y[te][mask],pp[mask])})
 for brand in sorted(df.iloc[te].brand_family.unique()):
  mask=(df.iloc[te].brand_family==brand).to_numpy();slices.append({'slice':brand,'n':int(mask.sum()),**metric(y[te][mask],pp[mask])})
 dump(OUT/'error_slices.json',slices);dump(OUT/'web_results.json',{'audit':audit,'metrics':json.loads((OUT/'metrics.json').read_text()),'split':json.loads((OUT/'split_policy.json').read_text()),'uncertainty':unc,'lobo':lobo,'videos':videos,'graphs':gs,'cases':cases,'new_video':new,'slices':slices,'graph_audit':json.loads((OUT/'graph_audit.json').read_text())})
 # Independent browser validation fixtures.
 ds=f.transform(df,gs,gnn=='Subgraph + similarity');allpred=predict(models[gnn],configs[gnn],ds);fixtures=[{'row':v,'graph':gg,'features':dd.x.tolist(),'edge_index':dd.edge_index.tolist(),'expected_log':float(pr)} for v,gg,dd,pr in zip(videos,gs,ds,allpred)];dump(OUT/'inference_fixture.json',fixtures)
 plt.rcParams.update({'font.family':'serif','font.size':8,'axes.spines.top':False,'axes.spines.right':False});fig,ax=plt.subplots(figsize=(3.4,2.25));names=list(results);ax.barh(names,[results[n]['log_RMSE'] for n in names],color=['#abb8b4' if n!=gnn else '#186b60' for n in names]);ax.set_xlabel('Test RMSE on ln(1 + views)');fig.tight_layout();fig.savefig(OUT/'comparison.pdf');fig.savefig(OUT/'comparison.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(3.4,2.25));ax.scatter(y[te],pp,c='#186b60',s=13,alpha=.75);lim=[min(y[te].min(),pp.min())-.3,max(y[te].max(),pp.max())+.3];ax.plot(lim,lim,'--',c='#999',lw=.7);ax.set(xlabel='Observed ln(1 + views)',ylabel='Predicted ln(1 + views)',xlim=lim,ylim=lim);fig.tight_layout();fig.savefig(OUT/'scatter.pdf');plt.close(fig)
 dump(OUT/'run_manifest.json',{'python':sys.executable,'torch':torch.__version__,'torch_geometric':__import__('torch_geometric').__version__,'source_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'snapshot_hash':hashlib.sha256(Path('data/youtube/snapshot.csv').read_bytes()).hexdigest(),'training_target':'MSE on training-standardized ln(1+views)','models':NAMES,'snapshot_selection':'Only fresh exact YouTube views; unavailable videos excluded, never approximated','known_limit':'Earlier v1 used archived labels; v2 is a new frozen snapshot and redesigned experiment, not an untouched external holdout. Hyperparameter tuning remains within v2 training only.'});print(json.dumps({'cohort':n,'split':[len(tr),len(te)],'primary':gnn,'test':results,'lobo':lobo},indent=2),flush=True)
if __name__=='__main__':main()
