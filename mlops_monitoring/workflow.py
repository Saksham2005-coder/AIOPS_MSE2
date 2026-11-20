import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [f"feature_{index}" for index in range(8)]
TARGET_COLUMN = "target"
ID_COLUMN = "record_id"
MINIMUM_F1 = 0.80
MAXIMUM_F1_DROP = 0.01


def _make_dataset(samples: int, random_state: int, prefix: str) -> pd.DataFrame:
    features, target = make_classification(
        n_samples=samples,
        n_features=len(FEATURE_COLUMNS),
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        weights=[0.58, 0.42],
        class_sep=2.0,
        flip_y=0.01,
        random_state=random_state,
    )
    frame = pd.DataFrame(features, columns=FEATURE_COLUMNS)
    frame[TARGET_COLUMN] = target
    frame.insert(0, ID_COLUMN, [f"{prefix}-{index:05d}" for index in range(samples)])
    return frame


def _build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=17)),
        ]
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _metrics(model: Pipeline, frame: pd.DataFrame) -> dict[str, float]:
    truth = frame[TARGET_COLUMN]
    predicted = model.predict(frame[FEATURE_COLUMNS])
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "f1": float(f1_score(truth, predicted, zero_division=0)),
    }


def validate_dataset(frame: pd.DataFrame, name: str) -> dict[str, Any]:
    required = [ID_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN]
    missing_columns = sorted(set(required) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"{name} is missing required columns: {missing_columns}")

    missing_values = frame[required].isna().sum()
    missing_counts = {column: int(count) for column, count in missing_values.items() if count}
    if missing_counts:
        raise ValueError(f"{name} contains missing values: {missing_counts}")
    if frame[ID_COLUMN].duplicated().any():
        raise ValueError(f"{name} contains duplicate {ID_COLUMN} values")
    if frame.duplicated(subset=[*FEATURE_COLUMNS, TARGET_COLUMN]).any():
        raise ValueError(f"{name} contains duplicate feature/target records")
    if frame.empty:
        raise ValueError(f"{name} must contain at least one record")
    if not set(frame[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError(f"{name} target values must be binary (0 or 1)")

    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "target_distribution": {
            str(label): int(count)
            for label, count in frame[TARGET_COLUMN].value_counts().sort_index().items()
        },
        "duplicate_ids": 0,
        "duplicate_records": 0,
        "missing_values": 0,
    }


def initialize(output_dir: Path) -> None:
    data_dir = output_dir / "data"
    artifacts_dir = output_dir / "artifacts"
    data_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    population = _make_dataset(2000, 31, "sample")
    training = population.iloc[:1200].copy()
    incoming = population.iloc[1200:1500].copy()
    holdout = population.iloc[1500:].copy()
    for prefix, frame in (
        ("train", training),
        ("batch", incoming),
        ("holdout", holdout),
    ):
        frame[ID_COLUMN] = [
            f"{prefix}-{index:05d}" for index in range(len(frame))
        ]
    training.to_csv(data_dir / "existing_training.csv", index=False)
    incoming.to_csv(data_dir / "incoming_batch.csv", index=False)
    holdout.to_csv(data_dir / "quality_gate_holdout.csv", index=False)

    model = _build_pipeline()
    model.fit(training[FEATURE_COLUMNS], training[TARGET_COLUMN])
    joblib.dump(model, artifacts_dir / "champion_model.joblib")
    print(f"Initialized synthetic datasets and champion model in {output_dir}")


def _track_mlflow(
    output_dir: Path,
    training_rows: int,
    holdout_rows: int,
    champion_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
) -> str:
    try:
        import mlflow
    except ImportError:
        return "not_installed"

    with mlflow.start_run(run_name="synthetic-candidate-retraining"):
        mlflow.log_param("training_rows", training_rows)
        mlflow.log_param("quality_gate_holdout_rows", holdout_rows)
        mlflow.log_metrics(
            {
                f"champion_holdout_{name}": value
                for name, value in champion_metrics.items()
            }
        )
        mlflow.log_metrics(
            {
                f"candidate_holdout_{name}": value
                for name, value in candidate_metrics.items()
            }
        )
        mlflow.log_artifact(str(output_dir / "artifacts" / "candidate_model.joblib"))
    return "logged"


def run_workflow(output_dir: Path) -> dict[str, Any]:
    data_dir = output_dir / "data"
    artifacts_dir = output_dir / "artifacts"
    reports_dir = output_dir / "reports"
    training = pd.read_csv(data_dir / "existing_training.csv")
    incoming = pd.read_csv(data_dir / "incoming_batch.csv")
    holdout = pd.read_csv(data_dir / "quality_gate_holdout.csv")
    batch_summary = validate_dataset(incoming, "incoming batch")
    validate_dataset(training, "existing training data")
    validate_dataset(holdout, "quality-gate holdout")

    champion_path = artifacts_dir / "champion_model.joblib"
    champion = joblib.load(champion_path)
    batch_metrics = _metrics(champion, incoming)
    evaluation_report = {
        "dataset": "incoming_batch.csv",
        "validation": batch_summary,
        "champion_metrics": batch_metrics,
    }
    evaluation_path = reports_dir / "evaluation_report.json"
    _write_json(evaluation_path, evaluation_report)

    combined_training = pd.concat([training, incoming], ignore_index=True)
    candidate = _build_pipeline()
    candidate.fit(
        combined_training[FEATURE_COLUMNS],
        combined_training[TARGET_COLUMN],
    )
    candidate_path = artifacts_dir / "candidate_model.joblib"
    joblib.dump(candidate, candidate_path)

    champion_holdout_metrics = _metrics(champion, holdout)
    candidate_holdout_metrics = _metrics(candidate, holdout)
    candidate_f1 = candidate_holdout_metrics["f1"]
    champion_f1 = champion_holdout_metrics["f1"]
    eligible = (
        candidate_f1 >= MINIMUM_F1
        and candidate_f1 >= champion_f1 - MAXIMUM_F1_DROP
    )
    decision = {
        "decision": "eligible_for_promotion" if eligible else "retain_champion",
        "promotion_automatic": False,
        "criterion": (
            f"candidate holdout F1 >= {MINIMUM_F1:.2f} and no more than "
            f"{MAXIMUM_F1_DROP:.2f} below champion holdout F1"
        ),
        "reason": (
            "Candidate meets the F1 quality gate; explicit human promotion is required."
            if eligible
            else "Candidate does not meet the F1 quality gate; retain the champion."
        ),
        "holdout_rows": int(len(holdout)),
        "champion_metrics": champion_holdout_metrics,
        "candidate_metrics": candidate_holdout_metrics,
        "candidate_model": str(candidate_path),
        "champion_model": str(champion_path),
        "mlflow_status": "pending",
    }
    decision_path = reports_dir / "quality_gate_decision.json"
    _write_json(decision_path, decision)
    decision["mlflow_status"] = _track_mlflow(
        output_dir,
        len(combined_training),
        len(holdout),
        champion_holdout_metrics,
        candidate_holdout_metrics,
    )
    _write_json(decision_path, decision)

    print(
        json.dumps(
            {
                "evaluation_report": str(evaluation_path),
                "quality_gate_decision": str(decision_path),
                "decision": decision["decision"],
                "mlflow_status": decision["mlflow_status"],
            },
            indent=2,
        )
    )
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["initialize", "run"])
    parser.add_argument("--output-dir", type=Path, default=Path("demo_output"))
    args = parser.parse_args()
    if args.command == "initialize":
        initialize(args.output_dir)
    else:
        run_workflow(args.output_dir)


if __name__ == "__main__":
    main()
