"""Verify trained inference parity and optional real Streamlit result switching."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def validate(prepared, check_app):
    from experiment_results import validate_bundle
    from model_service import ModelService

    folder = prepared / "evaluation" / "workspace" / "artifacts" / "v2"
    validate_bundle(folder)
    revised = ModelService(folder)
    fixtures = json.loads((folder / "inference_fixture.json").read_text(encoding="utf-8"))
    identities = [fixture["video_id"] for fixture in fixtures]
    if set(identities) != set(revised.videos) or len(identities) != len(revised.videos):
        raise ValueError("Inference fixtures do not cover the eligible cohort exactly once.")
    predictions = revised.predict_many(
        [revised.videos[key] for key in identities], [revised.graphs[key] for key in identities],
    )
    max_error = max(abs(pred["log"] - row["expected_log"]) for pred, row in zip(predictions, fixtures))
    if max_error > 1e-5:
        raise ValueError(f"App inference differs from training: maximum log error {max_error}")
    sample = dict(revised.videos[identities[0]])
    sample["main_message"] = "Ingredients like fresh eggs. Share happiness."
    brief = revised.query_graph(sample)
    message = next(node for node in brief["nodes"] if node["field"] == "main_message")
    if "like fresh eggs" not in message["text"] or "Share happiness" not in message["text"]:
        raise ValueError("New-brief inference reverted to the old text filter.")
    reference = revised.query_graph(sample, identities[0])
    revised.predict(sample, reference)
    revised.optimize(sample, reference)
    legacy = ModelService()
    old_fixture = json.loads(
        (ROOT / "artifacts" / "v2" / "inference_fixture.json").read_text(encoding="utf-8")
    )[0]
    old_prediction = legacy.predict(old_fixture["row"], old_fixture["graph"])["log"]
    if abs(old_prediction - old_fixture["expected_log"]) > 1e-5:
        raise ValueError("Legacy checkpoint predictions changed.")
    report = {
        "eligible_inference_checks": len(fixtures), "maximum_log_error": max_error,
        "legacy_inference_unchanged": True, "corrected_brief_filter": True,
        "prediction_and_optimization": True, "app_switching": False,
    }
    if check_app:
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
        if app.exception:
            raise ValueError(f"Original App failed: {[error.message for error in app.exception]}")
        app.selectbox(key="experiment_folder").select(str(folder)).run()
        if app.exception:
            raise ValueError(f"Revised App failed: {[error.message for error in app.exception]}")
        if not any("探索性修正实验" in warning.value for warning in app.warning):
            raise ValueError("The revised App did not show its exploratory warning.")
        expected = f'{len(revised.videos)}'
        if app.metric[0].value != expected:
            raise ValueError("The revised App is displaying a stale cohort size.")
        app.button[0].click().run()
        if app.exception:
            raise ValueError(f"App prediction failed: {[error.message for error in app.exception]}")
        if "last_prediction" not in app.session_state:
            raise ValueError("The prediction action did not produce an App result.")
        app.selectbox(key="experiment_folder").select(str(ROOT / "artifacts" / "v2")).run()
        if app.exception:
            raise ValueError(f"Switching back to the original failed: {[error.message for error in app.exception]}")
        if app.metric[0].value != str(len(legacy.videos)):
            raise ValueError("Switching back retained the revised cohort.")
        if "last_prediction" in app.session_state:
            raise ValueError("A prediction from the other experiment remained after switching.")
        report["app_switching"] = True
    (prepared / "evaluation" / "app_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--check-app", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate(args.input.resolve(), args.check_app), ensure_ascii=False))


if __name__ == "__main__":
    main()
