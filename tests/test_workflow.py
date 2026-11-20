import pandas as pd
import pytest

from mlops_monitoring.workflow import (
    FEATURE_COLUMNS,
    ID_COLUMN,
    TARGET_COLUMN,
    initialize,
    run_workflow,
    validate_dataset,
)


def _valid_frame() -> pd.DataFrame:
    row = {column: 0.0 for column in FEATURE_COLUMNS}
    row.update({ID_COLUMN: "record-1", TARGET_COLUMN: 0})
    return pd.DataFrame([row])


def test_validate_dataset_rejects_missing_values():
    frame = _valid_frame()
    frame.loc[0, FEATURE_COLUMNS[0]] = float("nan")

    with pytest.raises(ValueError, match="missing values"):
        validate_dataset(frame, "incoming batch")


def test_validate_dataset_rejects_duplicate_ids():
    frame = pd.concat([_valid_frame(), _valid_frame()], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate record_id"):
        validate_dataset(frame, "incoming batch")


def test_workflow_writes_reports_and_keeps_champion(tmp_path):
    initialize(tmp_path)
    champion_path = tmp_path / "artifacts" / "champion_model.joblib"
    champion_before = champion_path.read_bytes()

    decision = run_workflow(tmp_path)

    assert decision["decision"] == "eligible_for_promotion"
    assert decision["promotion_automatic"] is False
    assert decision["candidate_metrics"]["f1"] >= 0.80
    assert decision["candidate_metrics"]["f1"] >= (
        decision["champion_metrics"]["f1"] - 0.01
    )
    assert champion_path.read_bytes() == champion_before
    assert (tmp_path / "artifacts" / "candidate_model.joblib").is_file()
    assert (tmp_path / "reports" / "evaluation_report.json").is_file()
    assert (tmp_path / "reports" / "quality_gate_decision.json").is_file()
