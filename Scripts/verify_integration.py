"""Read-only saved-model regression checks; optional Streamlit widget checks."""
import argparse
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))


def verify(check_app=False):
    from model_service import ModelService
    from experiment_results import available_experiments

    service = ModelService()
    fixture = json.loads((ROOT / 'artifacts/v2/inference_fixture.json').read_text(encoding='utf-8'))
    predictions = service.predict_many([r['row'] for r in fixture], [r['graph'] for r in fixture])
    assert len(predictions) == len(fixture) == len(service.videos)
    error = max(abs(p['log'] - r['expected_log']) for p, r in zip(predictions, fixture))
    assert error < 1e-4, error
    for case in service.results['cases']:
        row = service.videos[case['video_id']]
        outcome = service.optimize(row, service.graphs[case['video_id']])
        assert abs(outcome['best']['log'] - case['optimized_log']) < 1e-4
        assert outcome['candidates'] == case['candidates_evaluated']
    errors = []
    experiments = available_experiments(errors=errors)
    result = {'fixture_videos': len(fixture), 'max_log_error': error,
              'optimization_cases': len(service.results['cases']),
              'available_experiments': len(experiments), 'optional_bundle_errors': errors,
              'app_checks': 'not_requested',
              'environment': {name: importlib.import_module(name).__version__ for name in
                              ['torch', 'torch_geometric', 'streamlit', 'numpy', 'pandas', 'sklearn']}}
    if check_app:
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=120).run()
        assert not app.exception, [e.message for e in app.exception]
        assert app.metric[0].value == str(len(service.videos))
        assert app.number_input(key='age___new__').value == service.age_support['training_median_days']
        app.number_input(key='age___new__').set_value(30.)
        app.button[0].click().run()
        assert not app.exception, [e.message for e in app.exception]
        saved = app.session_state['last_prediction']
        assert saved['result']['base']['age_diagnostic']['outside_training_range']
        assert 'views' not in saved['input']
        app.button[1].click().run()
        assert not app.exception, [e.message for e in app.exception]
        assert 'best' in app.session_state['last_prediction']['result']
        ref = fixture[0]['row']['video_id']
        app.selectbox(key='prediction_reference').set_value(ref).run()
        app.button[0].click().run()
        assert not app.exception, [e.message for e in app.exception]
        assert abs(app.session_state['last_prediction']['result']['base']['log'] - fixture[0]['expected_log']) < 1e-4
        app.toggle(key='graph_similarity').set_value(False).run()
        assert not app.exception, [e.message for e in app.exception]
        result['app_checks'] = 'passed_main_prediction_optimization_graph_and_age_warning'
        result['additional_experiment_switching'] = 'pending_actual_bundle' if len(experiments) == 1 else 'use validate_app_experiment.py for each bundle'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-app', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.check_app), ensure_ascii=False, indent=2))
