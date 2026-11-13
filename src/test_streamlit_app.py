"""Numerical regression and real Streamlit widget-flow tests for the final app."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'.project-deps'));sys.path.insert(0,str(ROOT/'src'))
from model_service import ModelService
from streamlit.testing.v1 import AppTest
s=ModelService();fixtures=json.loads((ROOT/'artifacts/v2/inference_fixture.json').read_text())
rows=[f['row'] for f in fixtures];graphs=[f['graph'] for f in fixtures]
predictions=s.predict_many(rows,graphs);error=max(abs(p['log']-f['expected_log']) for p,f in zip(predictions,fixtures));assert error<1e-4,error
for case in s.results['cases']:
 row=s.videos[case['video_id']];g=s.graphs[case['video_id']]
 assert s.query_graph(row,case['video_id'])==g
 o=s.optimize(row,g);assert abs(o['best']['log']-case['optimized_log'])<1e-4
 assert o['candidates']==case['candidates_evaluated']
 assert len(o['best']['changes'])<=2
row=next(r for r in rows if r['split']=='train').copy();row.update(title='New advertising brief',video_id='new-video-example',age_days=30.,duration_seconds=30.)
g=s.query_graph(row);assert all(n['type']!='evidence' for n in g['nodes']);p=s.predict(row,g);assert p['views']>=0 and p['age_diagnostic']['outside_training_range']
assert p['age_diagnostic']['input_age_days']==30 and s.age_support['training_min_days']>30
app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=40).run()
assert not app.exception,[e.message for e in app.exception]
app.selectbox(key='experiment_folder').set_value(str(ROOT/'artifacts/v2')).run()
assert app.number_input(key='age___new__').value==s.age_support['training_median_days']
app.number_input(key='age___new__').set_value(30.)
app.button[0].click().run()
assert any('范围外推' in w.value for w in app.warning)
assert app.session_state['last_prediction']['result']['base']['age_diagnostic']['outside_training_range']
assert not app.exception,[e.message for e in app.exception]
assert 'last_prediction' in app.session_state
next_prediction_new_input=dict(app.session_state['last_prediction']['input'])
assert app.session_state['last_prediction']['result']['base']['views']>=0
app.button[1].click().run()
assert not app.exception,[e.message for e in app.exception]
assert 'best' in app.session_state['last_prediction']['result']
ref=rows[0]['video_id'];app.selectbox(key='prediction_reference').set_value(ref).run()
assert not app.exception,[e.message for e in app.exception]
app.button[0].click().run()
actual=app.session_state['last_prediction']['result']['base']['log'];expected=next(f['expected_log'] for f in fixtures if f['row']['video_id']==ref)
assert abs(actual-expected)<1e-4,(actual,expected)
app.toggle(key='graph_similarity').set_value(False).run();assert not app.exception,[e.message for e in app.exception]
# A submitted creative edit must affect declared concept nodes and root inputs.
changed=next(v for v in s.features.root.enc.categories_[6] if v!=rows[0]['pacing'])
app.selectbox(key=f'edit_{ref}_pacing').set_value(changed)
app.button[1].click().run();assert not app.exception,[e.message for e in app.exception]
assert app.session_state['last_prediction']['input']['pacing']==changed
assert 'views' not in next_prediction_new_input
result={'framework':'Streamlit 1.66.1 + Python/PyG','fixture_videos':len(fixtures),'max_log_prediction_error':error,
        'optimization_cases':6,'new_brief_without_borrowed_evidence':True,'app_initialization':True,
        'new_video_prediction':True,'optimization_form':True,'existing_video_prediction_parity':True,
        '30_day_extrapolation_warning':True,'new_default_is_training_age_median':True,'graph_toggle':True,'edited_settings_submission':True,'app_exceptions':0}
(ROOT/'artifacts/v2/streamlit_validation.json').write_text(json.dumps(result,indent=2));print('PASS',result)
