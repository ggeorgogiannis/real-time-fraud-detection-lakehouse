import argparse
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from fraud_lakehouse.analytics import build_analytics_database
from fraud_lakehouse.ml_dataset import materialize_model_dataset
from fraud_lakehouse.optimization import (
    optimize_and_publish_hyperparameters,
)
from fraud_lakehouse.pipeline import run_batch_pipeline
from fraud_lakehouse.threshold_optimization import (
    optimize_and_publish_thresholds,
)
from fraud_lakehouse.training import train_and_publish_baselines

LOGGER = logging.getLogger(__name__)


def _parse_utc_datetime(value: str) -> datetime:
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected an ISO 8601 datetime, for example 2026-09-08T12:00:00+00:00."
        ) from exc

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("The datetime must include a timezone.")

    return parsed.astimezone(UTC)


def _add_log_level_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fraud-lakehouse",
        description="Run the local fraud-detection lakehouse pipeline.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Run the Bronze, Silver and Gold batch pipeline.",
    )
    run_parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Directory containing daily transaction files.",
    )
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root directory for generated lakehouse outputs.",
    )
    run_parser.add_argument(
        "--ingested-at-utc",
        type=_parse_utc_datetime,
        help="Optional timezone-aware ingestion timestamp.",
    )
    _add_log_level_argument(run_parser)

    analytics_parser = subparsers.add_parser(
        "build-analytics",
        help=("Create a DuckDB database over Silver and Gold Parquet files."),
    )
    analytics_parser.add_argument(
        "--silver-dir",
        type=Path,
        required=True,
        help="Directory containing Silver Parquet files.",
    )
    analytics_parser.add_argument(
        "--gold-dir",
        type=Path,
        required=True,
        help="Directory containing Gold Parquet files.",
    )
    analytics_parser.add_argument(
        "--database-path",
        type=Path,
        required=True,
        help="Path of the DuckDB database to create.",
    )
    _add_log_level_argument(analytics_parser)

    ml_dataset_parser = subparsers.add_parser(
        "build-ml-dataset",
        help=("Create chronological training, validation and test datasets."),
    )
    ml_dataset_parser.add_argument(
        "--transaction-features-path",
        type=Path,
        required=True,
        help="Path to the Gold transaction_features Parquet file.",
    )
    ml_dataset_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for generated model dataset artifacts.",
    )
    ml_dataset_parser.add_argument(
        "--train-end",
        type=_parse_utc_datetime,
        required=True,
        help="Exclusive UTC boundary for the training partition.",
    )
    ml_dataset_parser.add_argument(
        "--validation-end",
        type=_parse_utc_datetime,
        required=True,
        help="Exclusive UTC boundary for the validation partition.",
    )
    _add_log_level_argument(ml_dataset_parser)

    training_parser = subparsers.add_parser(
        "train-baselines",
        help="Train and evaluate baseline fraud classifiers.",
    )
    training_parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help=("Directory containing train, validation and test Parquet files."),
    )
    training_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for trained model artifacts and metrics.",
    )
    training_parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help=("Probability threshold used for binary classification metrics."),
    )
    _add_log_level_argument(training_parser)

    tuning_parser = subparsers.add_parser(
        "tune-hyperparameters",
        help=("Tune fraud classifiers using prequential temporal validation."),
    )
    tuning_parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Directory containing the training Parquet partition.",
    )
    tuning_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=("Directory for tuned model artifacts and search results."),
    )
    tuning_parser.add_argument(
        "--logistic-iterations",
        type=int,
        default=12,
        help=("Number of logistic-regression configurations to evaluate."),
    )
    tuning_parser.add_argument(
        "--xgboost-iterations",
        type=int,
        default=20,
        help=("Number of XGBoost configurations to evaluate."),
    )
    tuning_parser.add_argument(
        "--folds",
        type=int,
        default=3,
        help="Number of prequential assessment folds.",
    )
    tuning_parser.add_argument(
        "--assessment-days",
        type=int,
        default=14,
        help="Number of days in each assessment fold.",
    )
    tuning_parser.add_argument(
        "--gap-days",
        type=int,
        default=7,
        help="Label-delay gap before each assessment fold.",
    )
    tuning_parser.add_argument(
        "--card-precision-k",
        type=int,
        default=100,
        help=("Daily investigation capacity used for Card Precision at k."),
    )
    tuning_parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help=("Random seed used for reproducible parameter sampling."),
    )
    _add_log_level_argument(tuning_parser)

    threshold_parser = subparsers.add_parser(
        "optimize-thresholds",
        help=("Select capacity-constrained decision thresholds using validation data."),
    )
    threshold_parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help=("Directory containing the validation Parquet partition."),
    )
    threshold_parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directory containing tuned model artifacts.",
    )
    threshold_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the selected threshold policy.",
    )
    threshold_parser.add_argument(
        "--daily-card-capacity",
        type=int,
        default=100,
        help=("Maximum number of unique-card alerts allowed per day."),
    )
    _add_log_level_argument(threshold_parser)

    return parser


def _run_pipeline_command(arguments: argparse.Namespace) -> int:
    result = run_batch_pipeline(
        raw_dir=arguments.raw_dir,
        output_dir=arguments.output_dir,
        ingested_at_utc=arguments.ingested_at_utc,
    )

    LOGGER.info(
        ("pipeline_completed bronze_files=%d silver_files=%d quarantine_files=%d gold_tables=%d"),
        len(result.bronze_files),
        len(result.silver_files),
        len(result.quarantine_files),
        len(result.gold_tables),
    )

    return 0


def _build_analytics_command(arguments: argparse.Namespace) -> int:
    database_path = build_analytics_database(
        silver_dir=arguments.silver_dir,
        gold_dir=arguments.gold_dir,
        database_path=arguments.database_path,
    )

    LOGGER.info(
        "analytics_database_built database_path=%s",
        database_path,
    )

    return 0


def _build_ml_dataset_command(arguments: argparse.Namespace) -> int:
    outputs = materialize_model_dataset(
        transaction_features_path=arguments.transaction_features_path,
        output_dir=arguments.output_dir,
        train_end=arguments.train_end,
        validation_end=arguments.validation_end,
    )

    LOGGER.info(
        ("ml_dataset_built train_path=%s validation_path=%s test_path=%s metadata_path=%s"),
        outputs.train_path,
        outputs.validation_path,
        outputs.test_path,
        outputs.metadata_path,
    )

    return 0


def _train_baselines_command(arguments: argparse.Namespace) -> int:
    outputs = train_and_publish_baselines(
        dataset_dir=arguments.dataset_dir,
        output_dir=arguments.output_dir,
        threshold=arguments.threshold,
    )

    LOGGER.info(
        (
            "baseline_models_trained dummy_model_path=%s logistic_model_path=%s "
            "xgboost_model_path=%s metrics_path=%s"
        ),
        outputs.dummy_model_path,
        outputs.logistic_model_path,
        outputs.xgboost_model_path,
        outputs.metrics_path,
    )

    return 0


def _tune_hyperparameters_command(
    arguments: argparse.Namespace,
) -> int:
    outputs = optimize_and_publish_hyperparameters(
        dataset_dir=arguments.dataset_dir,
        output_dir=arguments.output_dir,
        logistic_iterations=arguments.logistic_iterations,
        xgboost_iterations=arguments.xgboost_iterations,
        n_folds=arguments.folds,
        assessment_days=arguments.assessment_days,
        gap_days=arguments.gap_days,
        card_precision_k=arguments.card_precision_k,
        random_state=arguments.random_state,
    )

    LOGGER.info(
        (
            "hyperparameter_optimization_completed "
            "logistic_model_path=%s xgboost_model_path=%s "
            "results_path=%s"
        ),
        outputs.logistic_model_path,
        outputs.xgboost_model_path,
        outputs.results_path,
    )

    return 0


def _optimize_thresholds_command(
    arguments: argparse.Namespace,
) -> int:
    outputs = optimize_and_publish_thresholds(
        dataset_dir=arguments.dataset_dir,
        model_dir=arguments.model_dir,
        output_dir=arguments.output_dir,
        daily_card_capacity=arguments.daily_card_capacity,
    )

    LOGGER.info(
        ("threshold_optimization_completed selected_model=%s selected_threshold=%s policy_path=%s"),
        outputs.selected_model_name,
        outputs.selected_threshold,
        outputs.policy_path,
    )

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, arguments.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    try:
        if arguments.command == "run":
            return _run_pipeline_command(arguments)

        if arguments.command == "build-analytics":
            return _build_analytics_command(arguments)

        if arguments.command == "build-ml-dataset":
            return _build_ml_dataset_command(arguments)
        if arguments.command == "train-baselines":
            return _train_baselines_command(arguments)
        if arguments.command == "tune-hyperparameters":
            return _tune_hyperparameters_command(arguments)
        if arguments.command == "optimize-thresholds":
            return _optimize_thresholds_command(arguments)
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error(
            "command_failed command=%s error=%s",
            arguments.command,
            exc,
        )
        return 1

    parser.error(f"Unsupported command: {arguments.command}")
