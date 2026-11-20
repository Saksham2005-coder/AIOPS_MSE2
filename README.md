# MLOps Model Monitoring and Retraining

A runnable MLOps practical assessment demo using a **synthetic binary
classification dataset**, a scikit-learn preprocessing/model pipeline, a
model-quality gate, and GitHub Actions CI. The workflow evaluates an existing
champion model, trains a separate candidate, and records a promotion decision
without automatically replacing the champion.

## Assessment tasks A–E

| Task | Implementation | Evidence |
| --- | --- | --- |
| **A — Prepare new data** | Generates a labeled incoming batch; validates required columns, missing values, duplicate IDs and duplicate feature/target rows; reports batch size and target distribution. | [`demo_output/data/incoming_batch.csv`](demo_output/data/incoming_batch.csv) and the validation section of the [evaluation report](demo_output/reports/evaluation_report.json) |
| **B — Evaluate existing model** | Loads the saved champion and predicts the incoming labeled batch. Reports accuracy, precision, recall, and F1. | [Evaluation report](demo_output/reports/evaluation_report.json) |
| **C — Retrain candidate** | Concatenates the original training data with the incoming batch and fits a new pipeline using the same `StandardScaler` + `LogisticRegression` preprocessing/model steps. Saves it separately. Logs parameters, holdout metrics, and the candidate artifact to MLflow when MLflow is installed. | [`demo_output/artifacts/candidate_model.joblib`](demo_output/artifacts/candidate_model.joblib) |
| **D — Apply quality gate** | Compares champion and candidate on a separate holdout that is not used for training or Task B evaluation. Records eligibility or champion retention; it never promotes automatically. | [Quality-gate decision](demo_output/reports/quality_gate_decision.json) |
| **E — Demonstrate and explain** | Includes generated datasets and models, machine-readable reports, these run instructions, automated tests, and a GitHub Actions workflow that uploads run evidence. | [`demo_output/`](demo_output/) and [GitHub Actions workflow](.github/workflows/model-monitoring.yml) |

### Quality-gate rationale

F1 is the primary quality metric because it balances precision and recall and
is more informative than accuracy alone when class counts differ. The
candidate must achieve a holdout F1 of at least **0.80** and may be at most
**0.01** below the champion. The 0.80 floor avoids promoting a weak model just
because it is close to a weak champion; the small non-inferiority tolerance
allows minor holdout variation without accepting a meaningful regression.
Accuracy, precision, and recall are also reported for context. An eligible
candidate still requires an explicit human promotion action.

## Run locally

Python 3.10 or later:

```bash
python -m pip install -e ".[test]"
python -m mlops_monitoring initialize --output-dir demo_output
python -m mlops_monitoring run --output-dir demo_output
python -m pytest
```

`initialize` is a one-time demo/bootstrap step. It generates one reproducible
2,000-row synthetic population and partitions it into 1,200 existing training
rows, 300 incoming labeled rows, and a separate 500-row quality-gate holdout.
It trains and saves the initial champion. `run` loads those existing files,
validates the datasets, evaluates the champion, retrains the candidate, and
writes the reports. Re-running `run` does not overwrite the champion.

The generated evidence is checked into `demo_output/`:

- `data/existing_training.csv`, `data/incoming_batch.csv`, and
  `data/quality_gate_holdout.csv`
- `artifacts/champion_model.joblib` and `artifacts/candidate_model.joblib`
- `reports/evaluation_report.json` and `reports/quality_gate_decision.json`

The reference execution evaluates the champion on the incoming batch with
**0.847 accuracy, 0.851 precision, 0.770 recall, and 0.808 F1**. On the
independent holdout, both champion and candidate have **0.839 F1**, so the
candidate is recorded as **eligible for promotion**. No automatic promotion
occurs.

### Optional MLflow tracking

MLflow is optional. If it is installed, the workflow tracks candidate training
row count, holdout size, champion and candidate holdout metrics, and the
candidate model artifact using the configured MLflow tracking URI. Otherwise,
local reports and model artifacts are still produced and the decision report
states `mlflow_status: not_installed`.

```bash
python -m pip install -e ".[mlflow]"
```

## Continuous integration

`.github/workflows/model-monitoring.yml` runs on pushes, pull requests, and
manual dispatch. It installs the project, generates the synthetic data and
champion, runs evaluation/retraining/quality gating, runs the test suite, and
uploads `demo_output/` as the `model-monitoring-evidence` workflow artifact.
