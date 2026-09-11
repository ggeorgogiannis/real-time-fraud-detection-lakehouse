import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest

from fraud_lakehouse import cli
from fraud_lakehouse.ml_dataset import ModelDatasetOutputs
from fraud_lakehouse.optimization import (
    HyperparameterOptimizationOutputs,
)
from fraud_lakehouse.pipeline import BatchPipelineResult
from fraud_lakehouse.threshold_optimization import (
    ThresholdOptimizationOutputs,
)
from fraud_lakehouse.training import BaselineTrainingOutputs


def test_main_runs_pipeline_and_logs_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    timestamp = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)

    def fake_run_batch_pipeline(
        raw_dir: Path,
        output_dir: Path,
        ingested_at_utc: datetime | None,
    ) -> BatchPipelineResult:
        assert raw_dir == tmp_path / "raw"
        assert output_dir == tmp_path / "processed"
        assert ingested_at_utc == timestamp

        return BatchPipelineResult(
            bronze_files=(output_dir / "bronze" / "daily.parquet",),
            silver_files=(output_dir / "silver" / "daily.parquet",),
            quarantine_files=(output_dir / "quarantine" / "daily.parquet",),
            gold_tables={
                "transaction_features": (output_dir / "gold" / "transaction_features.parquet")
            },
        )

    monkeypatch.setattr(cli, "run_batch_pipeline", fake_run_batch_pipeline)
    caplog.set_level(logging.INFO)

    exit_code = cli.main(
        [
            "run",
            "--raw-dir",
            str(raw_dir),
            "--output-dir",
            str(output_dir),
            "--ingested-at-utc",
            "2026-09-08T12:00:00Z",
        ]
    )

    assert exit_code == 0
    assert "pipeline_completed" in caplog.text


def test_main_builds_analytics_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    silver_dir = tmp_path / "silver"
    gold_dir = tmp_path / "gold"
    database_path = tmp_path / "analytics" / "fraud_lakehouse.duckdb"

    def fake_build_analytics_database(
        silver_dir: Path,
        gold_dir: Path,
        database_path: Path,
    ) -> Path:
        assert silver_dir == tmp_path / "silver"
        assert gold_dir == tmp_path / "gold"
        assert database_path == (tmp_path / "analytics" / "fraud_lakehouse.duckdb")
        return database_path

    monkeypatch.setattr(
        cli,
        "build_analytics_database",
        fake_build_analytics_database,
    )
    caplog.set_level(logging.INFO)

    exit_code = cli.main(
        [
            "build-analytics",
            "--silver-dir",
            str(silver_dir),
            "--gold-dir",
            str(gold_dir),
            "--database-path",
            str(database_path),
        ]
    )

    assert exit_code == 0
    assert "analytics_database_built" in caplog.text


def test_main_builds_ml_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    transaction_features_path = tmp_path / "gold" / "transaction_features.parquet"
    output_dir = tmp_path / "model"
    expected_train_end = datetime(2026, 1, 2, tzinfo=UTC)
    expected_validation_end = datetime(2026, 1, 3, tzinfo=UTC)

    def fake_materialize_model_dataset(
        transaction_features_path: Path,
        output_dir: Path,
        *,
        train_end: datetime,
        validation_end: datetime,
    ) -> ModelDatasetOutputs:
        assert transaction_features_path == (tmp_path / "gold" / "transaction_features.parquet")
        assert output_dir == tmp_path / "model"
        assert train_end == expected_train_end
        assert validation_end == expected_validation_end

        return ModelDatasetOutputs(
            train_path=output_dir / "train.parquet",
            validation_path=output_dir / "validation.parquet",
            test_path=output_dir / "test.parquet",
            metadata_path=output_dir / "dataset_metadata.json",
        )

    monkeypatch.setattr(
        cli,
        "materialize_model_dataset",
        fake_materialize_model_dataset,
    )
    caplog.set_level(logging.INFO)

    exit_code = cli.main(
        [
            "build-ml-dataset",
            "--transaction-features-path",
            str(transaction_features_path),
            "--output-dir",
            str(output_dir),
            "--train-end",
            "2026-01-02T00:00:00Z",
            "--validation-end",
            "2026-01-03T00:00:00Z",
        ]
    )

    assert exit_code == 0
    assert "ml_dataset_built" in caplog.text


def test_main_trains_baseline_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "models"

    def fake_train_and_publish_baselines(
        *,
        dataset_dir: Path,
        output_dir: Path,
        threshold: float,
    ) -> BaselineTrainingOutputs:
        assert dataset_dir == tmp_path / "dataset"
        assert output_dir == tmp_path / "models"
        assert threshold == 0.4

        return BaselineTrainingOutputs(
            dummy_model_path=output_dir / "dummy_prior.joblib",
            logistic_model_path=output_dir / "logistic_regression.joblib",
            xgboost_model_path=output_dir / "xgboost.joblib",
            metrics_path=output_dir / "metrics.json",
        )

    monkeypatch.setattr(
        cli,
        "train_and_publish_baselines",
        fake_train_and_publish_baselines,
    )
    caplog.set_level(logging.INFO)

    exit_code = cli.main(
        [
            "train-baselines",
            "--dataset-dir",
            str(dataset_dir),
            "--output-dir",
            str(output_dir),
            "--threshold",
            "0.4",
        ]
    )

    assert exit_code == 0
    assert "baseline_models_trained" in caplog.text
    assert f"xgboost_model_path={output_dir / 'xgboost.joblib'}" in caplog.text


def test_main_tunes_hyperparameters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "models"

    def fake_optimize_and_publish_hyperparameters(
        *,
        dataset_dir: Path,
        output_dir: Path,
        logistic_iterations: int,
        xgboost_iterations: int,
        n_folds: int,
        assessment_days: int,
        gap_days: int,
        card_precision_k: int,
        random_state: int,
    ) -> HyperparameterOptimizationOutputs:
        assert dataset_dir == tmp_path / "dataset"
        assert output_dir == tmp_path / "models"
        assert logistic_iterations == 3
        assert xgboost_iterations == 4
        assert n_folds == 2
        assert assessment_days == 5
        assert gap_days == 7
        assert card_precision_k == 100
        assert random_state == 17

        return HyperparameterOptimizationOutputs(
            logistic_model_path=(output_dir / "tuned_logistic_regression.joblib"),
            xgboost_model_path=output_dir / "tuned_xgboost.joblib",
            results_path=output_dir / "hyperparameter_search.json",
        )

    monkeypatch.setattr(
        cli,
        "optimize_and_publish_hyperparameters",
        fake_optimize_and_publish_hyperparameters,
    )
    caplog.set_level(logging.INFO)

    exit_code = cli.main(
        [
            "tune-hyperparameters",
            "--dataset-dir",
            str(dataset_dir),
            "--output-dir",
            str(output_dir),
            "--logistic-iterations",
            "3",
            "--xgboost-iterations",
            "4",
            "--folds",
            "2",
            "--assessment-days",
            "5",
            "--gap-days",
            "7",
            "--card-precision-k",
            "100",
            "--random-state",
            "17",
        ]
    )

    assert exit_code == 0
    assert "hyperparameter_optimization_completed" in caplog.text
    assert f"results_path={output_dir / 'hyperparameter_search.json'}" in caplog.text


def test_main_optimizes_decision_thresholds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    dataset_dir = tmp_path / "dataset"
    model_dir = tmp_path / "models"
    output_dir = tmp_path / "policy"

    def fake_optimize_and_publish_thresholds(
        *,
        dataset_dir: Path,
        model_dir: Path,
        output_dir: Path,
        daily_card_capacity: int,
    ) -> ThresholdOptimizationOutputs:
        assert dataset_dir == tmp_path / "dataset"
        assert model_dir == tmp_path / "models"
        assert output_dir == tmp_path / "policy"
        assert daily_card_capacity == 75

        return ThresholdOptimizationOutputs(
            policy_path=output_dir / "threshold_policy.json",
            selected_model_name="xgboost",
            selected_threshold=0.91,
        )

    monkeypatch.setattr(
        cli,
        "optimize_and_publish_thresholds",
        fake_optimize_and_publish_thresholds,
    )
    caplog.set_level(logging.INFO)

    exit_code = cli.main(
        [
            "optimize-thresholds",
            "--dataset-dir",
            str(dataset_dir),
            "--model-dir",
            str(model_dir),
            "--output-dir",
            str(output_dir),
            "--daily-card-capacity",
            "75",
        ]
    )

    assert exit_code == 0
    assert "threshold_optimization_completed" in caplog.text
    assert "selected_model=xgboost" in caplog.text
    assert "selected_threshold=0.91" in caplog.text
    assert f"policy_path={output_dir / 'threshold_policy.json'}" in caplog.text
