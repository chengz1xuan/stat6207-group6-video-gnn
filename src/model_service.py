"""Direct local inference for original and isolated exploratory experiments."""
from pathlib import Path
import sys,json,copy
import math,statistics
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.project-deps'))
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from revision import GraphFeatures,GraphRegressor,predict,edit_graph,new_graph,safe_text
from pipeline import CAT,TEXT,Features
import checkpoint_pickle
from experiment_results import validate_bundle
from Scripts.text_policy import response_free_text


class RevisedFeatures(Features):
    def text(self, data):
        return [
            ' '.join(response_free_text(row.get(field, '')) for field in TEXT)
            for _, row in data.iterrows()
        ]


class RevisedUnpickler(checkpoint_pickle.Unpickler):
    def find_class(self, module, name):
        if module in ('pipeline', '__main__') and name == 'Features':
            return RevisedFeatures
        return super().find_class(module, name)


class RevisedPickle:
    __name__ = 'revised_checkpoint_pickle'
    Unpickler = RevisedUnpickler
    load = staticmethod(checkpoint_pickle.load)

class ModelService:
    def __init__(self, folder=None):
        folder=Path(folder).resolve() if folder is not None else ROOT/'artifacts/v2'
        self.manifest=validate_bundle(folder)
        self.exploratory=self.manifest['kind']=='isolated_exploratory'
        self.folder=folder
        self.clean_text=response_free_text if self.exploratory else safe_text
        self.results=json.loads((folder/'web_results.json').read_text(encoding='utf-8'))
        self.scenarios=json.loads((folder/'optimization.json').read_text(encoding='utf-8'))
        # Trusted local training checkpoint only. Compatibility loader resolves
        # script-origin classes without changing Streamlit's __main__ module.
        cp=torch.load(folder/'primary_gnn.pt',map_location='cpu',weights_only=False,
                      pickle_module=RevisedPickle if self.exploratory else checkpoint_pickle)
        self.features=cp['features'];self.config=cp['config'];self.name=cp['model_name']
        self.model=GraphRegressor(cp['state_dict']['c1.lin_l.weight'].shape[1])
        self.model.load_state_dict(cp['state_dict']);self.model.eval()
        self.similarity=self.name=='Subgraph + similarity'
        self.band=self.results['metrics']['OOF_90_absolute_log_residual'][self.name]
        self.videos={r['video_id']:r for r in self.results['videos']}
        self.graphs={g['video_id']:g for g in self.results['graphs']}
        self.train_ids=set(cp['train_ids'])
        self.combos=self.scenarios['supported_combos']
        # Older exploratory exports lack age_support. Use training ages only.
        ages=[float(row['age_days']) for row in self.videos.values()
              if row['video_id'] in self.train_ids and row.get('age_days') is not None
              and math.isfinite(float(row['age_days']))]
        if not ages:
            raise ValueError('No finite training ages for inference diagnostics.')
        self.age_support={'training_min_days':min(ages),
            'training_median_days':statistics.median(ages),
            'training_max_days':max(ages),'training_n':len(ages),
            'target':'Cumulative snapshot views; age is a covariate, not a validated forecast horizon.'}
        if self.exploratory:
            expected_train={key for key,row in self.videos.items() if row['split']=='train'}
            expected_test={key for key,row in self.videos.items() if row['split']=='test'}
            if self.train_ids != expected_train or set(cp['test_ids']) != expected_test:
                raise ValueError('Checkpoint partitions differ from the displayed experiment.')
            if set(self.graphs) != set(self.videos):
                raise ValueError('Displayed video and graph inventories disagree.')
            if not isinstance(self.features.root, RevisedFeatures):
                raise ValueError('The revised inference text policy was not restored.')

    def predict_many(self, rows, graphs):
        ds=[self.features.graph(pd.Series(r),g,self.similarity)[0] for r,g in zip(rows,graphs)]
        logs=predict(self.model,self.config,ds)
        return [{'log':float(z),'views':float(np.expm1(np.clip(z,0,25))),
                 'band':[float(np.expm1(max(0,z-self.band))),float(np.expm1(min(25,z+self.band)))],
                 'age_diagnostic':self.age_diagnostic(row)} for z,row in zip(logs,rows)]

    def age_diagnostic(self,row):
        age=float(row['age_days']);a=self.age_support
        return {'input_age_days':age,
                'outside_training_range':not a['training_min_days']<=age<=a['training_max_days'],
                **a,'interpretation':'Exploratory cumulative-view estimate; not a validated fixed-horizon or 30-day forecast. Age within range alone does not validate a new brief.'}

    def predict(self,row,graph):
        return self.predict_many([row],[graph])[0]

    def query_graph(self, row, reference_id=None):
        if reference_id is None:
            graph=new_graph(pd.Series(row))
            if self.exploratory:
                for node in graph['nodes']:
                    if node['type']=='aspect':
                        node['text']=self.clean_text(row[node['field']])
            return graph
        g=copy.deepcopy(self.graphs[reference_id])
        # Update declared creative aspect nodes to match the edited form; retain
        # extracted evidence, whose consistency must be reviewed by the user.
        for n in g['nodes']:
            if n['type']=='aspect' and n['field'] in CAT+TEXT and row[n['field']]!=self.videos[reference_id][n['field']]:
                n['text']=self.clean_text(row[n['field']])
        return g

    def optimize(self,row,graph):
        fields=['creative_mechanic','structure_type','pacing']
        rows=[dict(row)];graphs=[graph];changes=[{}]
        for combo in self.combos:
            ch={k:v for k,v in zip(fields,combo) if row[k]!=v}
            if not 0<len(ch)<=2:continue
            r={**row,**ch};rows.append(r);graphs.append(edit_graph(graph,pd.Series(r),ch));changes.append(ch)
        predictions=self.predict_many(rows,graphs);j=int(np.argmax([p['log'] for p in predictions]))
        support=[r for r in self.videos.values() if r['video_id'] in self.train_ids and all(r[k]==rows[j][k] for k in fields)]
        return {'base':predictions[0],'best':{**predictions[j],'changes':changes[j],'row':rows[j],
                'support_examples':support[:3]},'candidates':len(rows)}

    def graph_dot(self,row,graph,similarity=False):
        data,added=self.features.graph(pd.Series(row),graph,similarity)
        quote=lambda x:json.dumps(str(x),ensure_ascii=False)
        lines=['graph G {','graph [bgcolor="transparent", overlap=false, rankdir=LR, nodesep=0.2];',
               'node [shape=box, style="rounded,filled", fontname="sans-serif", fontsize=10, color="#ccd8cc"];',
               'edge [color="#b3c5b7", penwidth=1];']
        for i,n in enumerate(graph['nodes']):
            label='VIDEO\n'+row.get('title','New brief')[:38] if i==0 else n['field']+'\n'+n['text'][:46]
            color={'video':'#186b60','aspect':'#ecd2a9','evidence':'#dce9dd'}[n['type']]
            lines.append(f'n{i} [label={quote(label)}, fillcolor="{color}", fontcolor="'+('white' if i==0 else '#1d3935')+'", tooltip='+quote(n['text'] or row.get('title',''))+'];')
        original={frozenset((e['source'],e['target'])):e for e in graph['edges']}
        for a,b in data.edge_index.T.tolist():
            if a>=b:continue
            edge=original.get(frozenset((graph['nodes'][a]['id'],graph['nodes'][b]['id'])))
            if edge:
                attrs='color="#97ad9c", tooltip='+quote(edge['type']+' | '+str(edge.get('source_edge_id') or 'declared input'))
            else:attrs='color="#bf8b42", style=dashed, tooltip="Heuristic semantic similarity"'
            lines.append(f'n{a} -- n{b} [{attrs}];')
        lines.append('}')
        return '\n'.join(lines),{'nodes':len(graph['nodes']),'source_relations':len(graph['edges']),
                                 'undirected_links':data.edge_index.shape[1]//2,'added_similarity_links':len({tuple(sorted(x)) for x in added})}
