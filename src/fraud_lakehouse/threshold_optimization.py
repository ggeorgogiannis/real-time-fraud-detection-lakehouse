import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import pandas as pd

from fraud_lakehouse.ml_dataset import (
    MODEL_FEATURE_COLUMNS,
    MODEL_TARGET_COLUMN,
)
from fraud_lakehouse.modeling import (
    BinaryClassificationMetrics,
    evaluate_binary_classifier,
    extract_model_inputs,
    fraud_probability,
)
from fraud_lakehouse.optimization import TunedModelArtifact
from fraud_lakehouse.thresholds import (
    CapacityConstrainedThreshold,
    ModelThresholdCandidate,
    select_capacity_constrained_threshold,
    select_model_threshold_policy,
)


@dataclass(frozen=True)
class ThresholdOptimizationOutputs:
    """Published operating policy selected from validation data."""

    policy_path: Path
    selected_model_name: str
    selected_threshold: float


def optimize_and_publish_thresholds(
    *,
    dataset_dir: Path,
    model_dir: Path,
    output_dir: Path,
    daily_card_capacity: int,
) -> ThresholdOptimizationOutputs:
    """Select and publish a capacity-constrained validation policy."""
    validation_path = Path(dataset_dir) / "validation.parquet"

    if not validation_path.is_file():
        raise FileNotFoundError(f"Validation dataset partition not found: {validation_path}")

    model_paths = {
        "logistic_regression": (Path(model_dir) / "tuned_logistic_regression.joblib"),
        "xgboost": Path(model_dir) / "tuned_xgboost.joblib",
    }

    for model_path in model_paths.values():
        if not model_path.is_file():
            raise FileNotFoundError(f"Tuned model artifact not found: {model_path}")

    validation_partition = pd.read_parquet(
        validation_path,
        engine="pyarrow",
    )
    validation_inputs = extract_model_inputs(validation_partition)

    candidates = []
    transaction_metrics_by_model = {}

    for model_name, model_path in model_paths.items():
        artifact = load_tuned_artifact(model_path)
        transformed_features = artifact.preprocessor.transform(validation_inputs.features)
        probabilities = fraud_probability(
            artifact.estimator,
            transformed_features,
        )
        threshold_metrics = select_capacity_constrained_threshold(
            validation_partition,
            probabilities,
            daily_card_capacity=daily_card_capacity,
        )
        transaction_metrics = evaluate_binary_classifier(
            target=validation_inputs.target,
            fraud_probability=probabilities,
            threshold=threshold_metrics.threshold,
        )

        candidates.append(
            ModelThresholdCandidate(
                model_name=model_name,
                average_precision=transaction_metrics.average_precision,
                threshold_metrics=threshold_metrics,
            )
        )
        transaction_metrics_by_model[model_name] = transaction_metrics

    selected = select_model_threshold_policy(candidates)

    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    policy_path = output_directory / "threshold_policy.json"
    temporary_policy_path = output_directory / ".threshold_policy.json.tmp"

    policy = _policy_payload(
        candidates=candidates,
        transaction_metrics_by_model=transaction_metrics_by_model,
        selected=selected,
        model_paths=model_paths,
        daily_card_capacity=daily_card_capacity,
    )

    try:
        temporary_policy_path.write_text(
            json.dumps(
                policy,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary_policy_path.replace(policy_path)
    finally:
        temporary_policy_path.unlink(missing_ok=True)

    return ThresholdOptimizationOutputs(
        policy_path=policy_path,
        selected_model_name=selected.model_name,
        selected_threshold=selected.threshold_metrics.threshold,
    )


def load_tuned_artifact(
    model_path: Path,
) -> TunedModelArtifact:
    artifact = joblib.load(model_path)

    if not isinstance(artifact, TunedModelArtifact):
        raise TypeError(f"Unexpected tuned model artifact type: {model_path}")

    if artifact.feature_columns != MODEL_FEATURE_COLUMNS:
        raise ValueError(f"Unexpected feature contract in tuned model: {model_path}")

    if artifact.target_column != MODEL_TARGET_COLUMN:
        raise ValueError(f"Unexpected target contract in tuned model: {model_path}")

    return artifact


def _policy_payload(
    *,
    candidates: list[ModelThresholdCandidate],
    transaction_metrics_by_model: dict[
        str,
        BinaryClassificationMetrics,
    ],
    selected: ModelThresholdCandidate,
    model_paths: dict[str, Path],
    daily_card_capacity: int,
) -> dict[str, object]:
    models = {
        candidate.model_name: _model_policy_payload(
            candidate=candidate,
            transaction_metrics=transaction_metrics_by_model[candidate.model_name],
            model_path=model_paths[candidate.model_name],
        )
        for candidate in candidates
    }

    return {
        "schema_version": 1,
        "selection_data": "validation",
        "selection_objective": "maximize_card_day_recall",
        "selection_order": [
            "card_day_recall",
            "card_day_precision",
            "average_precision",
        ],
        "operating_constraint": {
            "daily_card_capacity": daily_card_capacity,
            "scope": "unique_customer_cards_per_day",
        },
        "test_evaluation": {
            "performed": False,
        },
        "selected_policy": {
            "model_name": selected.model_name,
            "model_artifact": model_paths[selected.model_name].name,
            "threshold": selected.threshold_metrics.threshold,
        },
        "models": models,
    }


def _model_policy_payload(
    *,
    candidate: ModelThresholdCandidate,
    transaction_metrics: BinaryClassificationMetrics,
    model_path: Path,
) -> dict[str, object]:
    threshold_metrics: CapacityConstrainedThreshold = candidate.threshold_metrics

    return {
        "model_artifact": model_path.name,
        "threshold": threshold_metrics.threshold,
        "average_precision": candidate.average_precision,
        "transaction_metrics": asdict(transaction_metrics),
        "card_day_metrics": asdict(threshold_metrics),
    }
