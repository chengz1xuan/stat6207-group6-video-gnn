"""Post-audit diagnostics and report figures from frozen corrected predictions.

No fitting or model selection occurs here. The displayed graph is an induced
subset of an actual training graph, chosen in inventory order without labels.
"""
from pathlib import Path
import os,json,textwrap
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/matplotlib'))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=ROOT/'artifacts/v2'

def make_audit():
    load=lambda name:json.loads((D/name).read_text())
    save=lambda name,obj:(D/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False))
    df=pd.read_csv(ROOT/'data/processed/videos_v2.csv');splits=pd.read_csv(D/'splits.csv')
    m=load('metrics.json');name=m['selected_primary_gnn'];p=pd.read_csv(D/'test_predictions.csv')
    y=np.log1p(p.views.to_numpy(float));groups=p.group.to_numpy();unique=np.unique(groups)
    pairs=[(name,'Metadata Ridge'),(name,'Pooled Ridge'),('Node structural','Node no-brand edges'),('Node structural','Brand mean')]
    draws={a+' minus '+b:[] for a,b in pairs};rng=np.random.default_rng(44)
    for _ in range(2000):
        ix=np.concatenate([np.flatnonzero(groups==g) for g in rng.choice(unique,len(unique),replace=True)])
        for a,b in pairs:
            rmse=lambda n:float(np.mean((p[n+'_log'].to_numpy()[ix]-y[ix])**2)**.5)
            draws[a+' minus '+b].append(rmse(a)-rmse(b))
    comparison={a+' minus '+b:{'delta_test_log_RMSE':m['test'][a]['log_RMSE']-m['test'][b]['log_RMSE'],
                    'campaign_bootstrap_95':np.percentile(draws[a+' minus '+b],[2.5,97.5]).tolist()}
                for a,b in pairs}
    save('paired_comparisons.json',{'metric':'RMSE difference: first minus second; negative favors first','draws':2000,'comparisons':comparison,
         'limitation':'Conditional, exploratory post-audit comparisons; neither causal identification nor independent replication.'})
    previous=load('reupload_error_audit.json');vid=previous['video_id']
    fixture=next(f for f in load('inference_fixture.json') if f['row']['video_id']==vid)
    previous.update(corrected_split=fixture['row']['split'],corrected_model=name,
                    corrected_predicted_views=float(np.expm1(np.clip(fixture['expected_log'],0,25))),
                    corrected_prediction_status='In-sample fit diagnostic, not a held-out error' if fixture['row']['split']=='train' else 'Corrected held-out prediction')
    save('reupload_error_audit.json',previous)
    errors=abs(p[name+'_log'].to_numpy()-y);worst=p.iloc[int(np.argmax(errors))]
    save('corrected_largest_error.json',{'video_id':worst.video_id,'title':worst.title,'observed_views':int(worst.views),
          'predicted_views':float(worst[name+'_views']),'absolute_log_error':float(max(errors)),
          'channel_name':df.set_index('video_id').loc[worst.video_id,'channel_name'],'split':'test','model':name})
    plt.rcParams.update({'font.family':'serif','font.size':8,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(3.45,2.35));pp=p[name+'_log'].to_numpy()
    ax.scatter(y,pp,c='#186b60',s=13,alpha=.8,zorder=2)
    j=int(np.argmax(errors));ax.scatter([y[j]],[pp[j]],s=34,facecolors='none',edgecolors='#b85836',zorder=3)
    ax.annotate('Largest corrected\ntest log error',(y[j],pp[j]),xytext=(8,12),textcoords='offset points',fontsize=6)
    lim=[min(y.min(),pp.min())-.5,max(y.max(),pp.max())+.5];ax.plot(lim,lim,'--',color='#999',lw=.8)
    ax.set(xlabel='Observed ln(1 + views)',ylabel='Predicted ln(1 + views)',xlim=lim,ylim=lim)
    fig.tight_layout(pad=.6);fig.savefig(D/'scatter.pdf');fig.savefig(D/'scatter.png',dpi=220);plt.close(fig)
    # Fixed inventory-order selection, independent of response or model error.
    train=set(splits.loc[splits.split=='train','video_id']);graphs=load('raw_graphs.json')
    wanted=['brand_family','creative_mechanic','structure_type','pacing','cta_explicitness','proof']
    graph=next(g for g in graphs if g['video_id'] in train and sum(n['field'] in wanted for n in g['nodes'])>=5 and any(n['type']=='evidence' for n in g['nodes']))
    aspects=[]
    for field in wanted:
        found=[n for n in graph['nodes'] if n['type']=='aspect' and n['field']==field]
        if found:aspects.append(found[0])
    selected={n['id'] for n in aspects}
    linked={e['target'] for e in graph['edges'] if e['type']=='aspect_supported_by_evidence' and e['source'] in selected}
    evidence=sorted([n for n in graph['nodes'] if n['type']=='evidence'],key=lambda n:(n['id'] not in linked,n['id']))[:2]
    nodes=[graph['nodes'][0]]+aspects+evidence;ids={n['id'] for n in nodes}
    edges=[e for e in graph['edges'] if e['source'] in ids and e['target'] in ids]
    save('report_graph.json',{'video_id':graph['video_id'],'source_package':graph['package'],'selection':'First training graph in inventory order with >=5 selected aspects and evidence; induced display subset; labels not used.','full_node_count':len(graph['nodes']),'nodes':nodes,'edges':edges,'note':'Node boxes show source field/value excerpts; evidence text omitted for legibility. Full original text and IDs are retained here and in raw_graphs.json.'})
    fig,ax=plt.subplots(figsize=(3.45,2.5));ax.axis('off');ax.set(xlim=(-.04,1.04),ylim=(-.08,1.08))
    root=nodes[0];pos={root['id']:(.1,.5)}
    for i,n in enumerate(aspects):pos[n['id']]=(.52,1-i/(max(1,len(aspects)-1)))
    for i,n in enumerate(evidence):pos[n['id']]=(.95,.72-i*.44)
    for e in edges:
        a,b=pos[e['source']],pos[e['target']];support=e['type']=='aspect_supported_by_evidence'
        ax.plot([a[0],b[0]],[a[1],b[1]],color='#bd914b' if support else '#9eb5aa',lw=.8,linestyle='--' if e.get('heuristic') else '-',zorder=1)
    for i,n in enumerate(nodes):
        x,z=pos[n['id']];kind=n['type']
        if kind=='video':label='VIDEO\n(context root)'
        elif kind=='evidence':label='Evidence '+str(evidence.index(n)+1)+'\n'+n['field'].replace('extract_','').replace('_','\n')
        else:
            value=n['text'];value=value if value.isascii() else '[original text]'
            label=n['field'].replace('_',' ')+'\n'+'\n'.join(textwrap.wrap(value[:35],19))
        ax.text(x,z,label,ha='center',va='center',fontsize=5.6,color='white' if kind=='video' else '#233d36',
                bbox=dict(boxstyle='round,pad=.3',facecolor={'video':'#186b60','aspect':'#f1dcb9','evidence':'#dfebe2'}[kind],edgecolor='#aac1b4',lw=.6),zorder=2)
    fig.subplots_adjust(left=.02,right=.98,bottom=.03,top=.97);fig.savefig(D/'concept_graph.pdf');fig.savefig(D/'concept_graph.png',dpi=240);plt.close(fig)
    print('Saved corrected-error audit, paired comparisons and two measured report figures.')

if __name__=='__main__':make_audit()
