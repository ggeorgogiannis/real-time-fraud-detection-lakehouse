import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from fraud_lakehouse.ml_dataset import (
    MODEL_FEATURE_COLUMNS,
    MODEL_TARGET_COLUMN,
)
from fraud_lakehouse.modeling import (
    BaselineEstimator,
    prepare_model_partitions,
    train_baseline_models,
)


@dataclass(frozen=True)
class BaselineModelArtifact:
    """Serializable preprocessing and estimator bundle."""

    preprocessor: Pipeline
    estimator: BaselineEstimator
    feature_columns: tuple[str, ...]
    target_column: str
    threshold: float


@dataclass(frozen=True)
class BaselineTrainingOutputs:
    """Paths to published baseline model artifacts and metrics."""

    dummy_model_path: Path
    logistic_model_path: Path
    metrics_path: Path


def train_and_publish_baselines(
    *,
    dataset_dir: Path,
    output_dir: Path,
    threshold: float = 0.5,
) -> BaselineTrainingOutputs:
    """Train baseline models from temporal partitions and publish results."""
    dataset_directory = Path(dataset_dir)
    partition_paths = {
        name: dataset_directory / f"{name}.parquet" for name in ("train", "validation", "test")
    }

    for partition_path in partition_paths.values():
        if not partition_path.is_file():
            raise FileNotFoundError(f"Model dataset partition not found: {partition_path}")

    partitions = {
        name: pd.read_parquet(path, engine="pyarrow") for name, path in partition_paths.items()
    }

    prepared = prepare_model_partitions(
        train=partitions["train"],
        validation=partitions["validation"],
        test=partitions["test"],
    )
    evaluations = train_baseline_models(
        prepared,
        threshold=threshold,
    )

    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    model_paths = {
        evaluation.name: output_directory / f"{evaluation.name}.joblib"
        for evaluation in evaluations
    }
    temporary_model_paths = {name: output_directory / f".{name}.joblib.tmp" for name in model_paths}

    metrics_path = output_directory / "metrics.json"
    temporary_metrics_path = output_directory / ".metrics.json.tmp"

    metrics = {
        "schema_version": 1,
        "primary_metric": "average_precision",
        "threshold": float(threshold),
        "library": {
            "name": "scikit-learn",
            "version": sklearn.__version__,
        },
        "models": {
            evaluation.name: {
                "validation": asdict(evaluation.validation_metrics),
                "test": asdict(evaluation.test_metrics),
            }
            for evaluation in evaluations
        },
    }

    try:
        for evaluation in evaluations:
            artifact = BaselineModelArtifact(
                preprocessor=prepared.preprocessor,
                estimator=evaluation.estimator,
                feature_columns=MODEL_FEATURE_COLUMNS,
                target_column=MODEL_TARGET_COLUMN,
                threshold=float(threshold),
            )
            joblib.dump(
                artifact,
                temporary_model_paths[evaluation.name],
            )

        temporary_metrics_path.write_text(
            json.dumps(
                metrics,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

        for name, model_path in model_paths.items():
            temporary_model_paths[name].replace(model_path)

        # Publish metrics last so they describe a complete artifact set.
        temporary_metrics_path.replace(metrics_path)
    finally:
        for temporary_path in (
            *temporary_model_paths.values(),
            temporary_metrics_path,
        ):
            temporary_path.unlink(missing_ok=True)

    return BaselineTrainingOutputs(
        dummy_model_path=model_paths["dummy_prior"],
        logistic_model_path=model_paths["logistic_regression"],
        metrics_path=metrics_path,
    )
