"""Create auditable report diagnostics/figures and an explicitly author-declared brief.
No fitting, split selection or modification of frozen training results occurs here.
"""
from pathlib import Path
import sys,os,json,hashlib,textwrap,collections
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'.project-deps'))
os.environ.setdefault('MPLCONFIGDIR',str(R/'tmp/matplotlib'))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from model_service import ModelService
from training_measurements import export


def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
    config=json.loads((R/'artifacts/final_v4.json').read_text());D=R/config['primary_folder'];service=ModelService(D)
    export(D);df=pd.DataFrame(service.results['videos']);pred=pd.read_csv(D/'test_predictions.csv')
    name=service.name;y=np.log1p(pred.views.to_numpy(float));p=pred[name+'_log'].to_numpy();groups=pred.group.to_numpy();unique=np.unique(groups)
    pairs=[(name,'Metadata Ridge'),('Node structural','Node no-brand edges')]
    samples={a+' minus '+b:[] for a,b in pairs};rng=np.random.default_rng(44)
    for _ in range(2000):
        ix=np.concatenate([np.flatnonzero(groups==g) for g in rng.choice(unique,len(unique),replace=True)])
        rmse=lambda n:float(np.mean((pred[n+'_log'].to_numpy()[ix]-y[ix])**2)**.5)
        for a,b in pairs:samples[a+' minus '+b].append(rmse(a)-rmse(b))
    comparisons={k:{'campaign_bootstrap_95':np.percentile(v,[2.5,97.5]).tolist()} for k,v in samples.items()}
    # Drafted as a hypothetical storyboard, never described as extracted evidence.
    template=next(v for v in service.videos.values() if v['video_id'] in service.train_ids and v['brand_family']=='MTR')
    fields=['brand_family','creative_mechanic','structure_type','evidence_mode','cta_explicitness','proof_strength','pacing','production_complexity']
    row={k:template[k] for k in fields};row.update(video_id='author-declared-mtr-brief',title='Proposed cross-boundary rail brief',
        duration_seconds=30.,age_days=service.age_support['training_median_days'],
        main_message='Present the East Rail Line as a rail option for a Hong Kong to Shenzhen journey.',
        hook='Planning a cross-boundary journey? Start with a clear route.',
        value_promise='A route-map demonstration can explain a rail option; check current travel information before production.',
        cta='Check the current MTR route and timetable before travelling.')
    outcome=service.optimize(row,service.query_graph(row));best=outcome['best']
    support=[{k:r[k] for k in ['video_id','youtube_id','title','source_url','brand_family']} for r in best['support_examples']]
    show={'row':row,'video_id':row['video_id'],'title':row['title'],'brand':row['brand_family'],
        'brief_objective':'Explore a clearly explained Hong Kong--Shenzhen rail option; all wording is a proposed brief, not verified current service information.',
        'creative_elements':{k:row[k] for k in ['creative_mechanic','structure_type','pacing','main_message','hook','cta']},
        'base_prediction':outcome['base']['views'],'optimized_prediction':best['views'],'base_log':outcome['base']['log'],'optimized_log':best['log'],
        'uncertainty_band':outcome['base']['band'],'optimized_uncertainty_band':best['band'],
        'changes':best['changes'],'support_examples':support,'candidates_evaluated':outcome['candidates'],
        'age_days':row['age_days'],'observed_views':None,'input_provenance':'Author-declared hypothetical brief; no borrowed extracted evidence, no realized video outcome.',
        'checkpoint_sha256':hashlib.sha256((D/'primary_gnn.pt').read_bytes()).hexdigest(),
        'storyboard':[{'seconds':'0-3','purpose':'Hook','content':row['hook']},
          {'seconds':'3-20','purpose':'Explain the travel need and an option','content':'Show a hypothetical journey-planning need and an East Rail route-map option.'},
          {'seconds':'20-27','purpose':'Apply proposed settings','content':'; '.join(k+' = '+v for k,v in best['changes'].items()) or 'Keep supported baseline settings.'},
          {'seconds':'27-30','purpose':'CTA','content':row['cta']}],
        'transferable_takeaway':'Test a coherent rewritten storyboard under comparable exposure; categorical search and training support do not establish causal uplift.'}
    showpath=R/'artifacts/final_v4_showcase.json';dump(showpath,show)
    config.update(showcase_path=str(showpath.relative_to(R)),showcase_sha256=hashlib.sha256(showpath.read_bytes()).hexdigest())
    dump(R/'artifacts/final_v4.json',config)
    plt.rcParams.update({'font.family':'serif','font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(3.45,2.3));ax.scatter(y,p,s=15,color='#186b60',alpha=.85,label='Primary GNN')
    meta=pred['Metadata Ridge_log'].to_numpy();ax.scatter(y,meta,s=14,marker='x',color='#9c9c9c',alpha=.6,label='Metadata Ridge')
    lim=[min(y.min(),p.min(),meta.min())-.4,max(y.max(),p.max(),meta.max())+.4];ax.plot(lim,lim,'--',color='#999',lw=.7)
    j=int(np.argmax(abs(y-p)));ax.scatter(y[j],p[j],facecolors='none',edgecolors='#bc5a3d',s=44,zorder=3)
    ax.set(xlabel='Observed ln(1 + views)',ylabel='Predicted ln(1 + views)',xlim=lim,ylim=lim);ax.legend(fontsize=6,loc='lower right',frameon=False)
    fig.tight_layout(pad=.6);fig.savefig(D/'report_scatter.pdf');fig.savefig(D/'report_scatter.png',dpi=220);plt.close(fig)
    # A real training graph with a surviving source support and real semantic links.
    candidates=[]
    for g in service.graphs.values():
        if g['video_id'] not in service.train_ids:continue
        support_edges=[e for e in g['edges'] if e['type']=='aspect_supported_by_evidence']
        if support_edges:
            _,added=service.features.graph(pd.Series(service.videos[g['video_id']]),g,True)
            if added:candidates.append((g,support_edges[0],added));break
    if not candidates:raise ValueError('No training graph with both surviving support and similarity extension.')
    g,support_edge,added=candidates[0];ids=[g['nodes'][0]['id'],support_edge['source'],support_edge['target']]
    a,b=added[0]
    for index in [a,b]:
        if g['nodes'][index]['id'] not in ids:ids.append(g['nodes'][index]['id'])
    for field in ['brand_family','creative_mechanic','structure_type','pacing','cta_explicitness']:
        node=next((n for n in g['nodes'] if n['field']==field),None)
        if node and node['id'] not in ids and len(ids)<9:ids.append(node['id'])
    nodes=[next(n for n in g['nodes'] if n['id']==key) for key in ids]
    edges=[e for e in g['edges'] if e['source'] in ids and e['target'] in ids]
    semantic=sorted({tuple(sorted((g['nodes'][a]['id'],g['nodes'][b]['id']))) for a,b in added if g['nodes'][a]['id'] in ids and g['nodes'][b]['id'] in ids})
    G=nx.Graph();G.add_nodes_from(ids);G.add_edges_from((e['source'],e['target']) for e in edges);G.add_edges_from(semantic)
    pos={ids[0]:(.06,.5)}
    aspects=[n for n in nodes if n['type']=='aspect'];evidences=[n for n in nodes if n['type']=='evidence']
    for n,z in zip(aspects,np.linspace(.96,.08,len(aspects))):pos[n['id']]=(.5,float(z))
    for n,z in zip(evidences,np.linspace(.75,.25,len(evidences))):pos[n['id']]=(.95,float(z))
    fig,ax=plt.subplots(figsize=(3.45,2.6));ax.axis('off')
    for e in edges:
        supported=e['type']=='aspect_supported_by_evidence';nx.draw_networkx_edges(G,pos,edgelist=[(e['source'],e['target'])],ax=ax,width=1.1,edge_color='#427fa0' if supported else '#a3b7ac')
    nx.draw_networkx_edges(G,pos,edgelist=semantic,ax=ax,width=.8,edge_color='#bf8b42',style='dashed')
    for n in nodes:
        kind=n['type'];text=n['text']
        if kind=='video':label='VIDEO'
        elif kind=='evidence':label='Evidence '+str(evidences.index(n)+1)+'\n'+n['field'].replace('extract_','').replace('_',' ')
        else:label=n['field'].replace('_',' ')+'\n'+(' '.join(text.split())[:35] if text.isascii() else '[source text]')
        label='\n'.join(textwrap.fill(line,19) for line in label.split('\n'));x,z=pos[n['id']]
        ax.text(x,z,label,ha='center',va='center',fontsize=5.8,color='white' if kind=='video' else '#233d36',bbox=dict(boxstyle='round,pad=.3',facecolor={'video':'#186b60','aspect':'#efd9b5','evidence':'#dfebe2'}[kind],edgecolor='#a3b7ac',lw=.6),zorder=3)
    ax.set_xlim(-.12,1.15);ax.set_ylim(-.1,1.12);fig.subplots_adjust(left=.02,right=.98,bottom=.02,top=.98)
    fig.savefig(D/'report_graph.pdf');fig.savefig(D/'report_graph.png',dpi=240);plt.close(fig)
    diagnostics={'campaign_groups_test':len(unique),'paired_RMSE_comparisons':comparisons,
        'display_graph':{'video_id':g['video_id'],'youtube_id':service.videos[g['video_id']]['youtube_id'],'nodes':nodes,'source_edges':edges,'semantic_edges':semantic,'full_node_count':len(g['nodes']),'selection':'First training graph in inventory order with surviving deterministic support and semantic extension; no response-based selection.'},
        'showcase':{'path':str(showpath.relative_to(R)),'sha256':config['showcase_sha256']},'finalizer_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    dump(D/'report_diagnostics.json',diagnostics)
    print('Created bound author-declared brief, full training measurements, paired intervals and real report figures.')


if __name__=='__main__':main()
