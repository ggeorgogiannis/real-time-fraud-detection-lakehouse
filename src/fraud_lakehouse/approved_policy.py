import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fraud_lakehouse.modeling import fraud_probability
from fraud_lakehouse.optimization import TunedModelArtifact
from fraud_lakehouse.threshold_optimization import load_tuned_artifact

CAPACITY_SCOPE = "unique_customer_cards_per_day"


@dataclass(frozen=True)
class ApprovedPolicyMetadata:
    """Validated metadata for the locked fraud-detection policy."""

    model_name: str
    model_artifact_name: str
    threshold: float
    daily_card_capacity: int
    capacity_scope: str
    policy_source: str
    evaluation_source: str
    validation_transaction_metrics: dict[str, object]
    validation_card_day_metrics: dict[str, object]
    test_transaction_metrics: dict[str, object]
    test_card_day_metrics: dict[str, object]


@dataclass(frozen=True)
class ApprovedFraudPolicy:
    """Self-contained fitted model and its approved operating policy."""

    model: TunedModelArtifact
    metadata: ApprovedPolicyMetadata

    def predict(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Score model features and apply the locked decision threshold."""
        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame.")

        if features.empty:
            raise ValueError("features must contain at least one row.")

        missing_columns = sorted(set(self.model.feature_columns).difference(features.columns))
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"Missing required model features: {missing}")

        ordered_features = features.loc[
            :,
            list(self.model.feature_columns),
        ]
        transformed_features = self.model.preprocessor.transform(ordered_features)
        probabilities = fraud_probability(
            self.model.estimator,
            transformed_features,
        )

        return pd.DataFrame(
            {
                "fraud_probability": probabilities,
                "threshold_decision": (probabilities >= self.metadata.threshold),
            },
            index=features.index,
        )


def load_approved_fraud_policy(
    *,
    model_dir: Path,
    policy_path: Path,
    evaluation_path: Path,
) -> ApprovedFraudPolicy:
    """Load and cross-check the model, policy and final evaluation."""
    model_directory = Path(model_dir)
    locked_policy_path = Path(policy_path)
    final_evaluation_path = Path(evaluation_path)

    policy = _load_json_object(
        locked_policy_path,
        description="Threshold policy",
    )
    evaluation = _load_json_object(
        final_evaluation_path,
        description="Final evaluation",
    )

    metadata = _validate_approval_contract(
        policy=policy,
        evaluation=evaluation,
        policy_path=locked_policy_path,
        evaluation_path=final_evaluation_path,
    )

    model_path = model_directory / metadata.model_artifact_name
    if not model_path.is_file():
        raise FileNotFoundError(f"Approved model artifact not found: {model_path}")

    model = load_tuned_artifact(model_path)

    return ApprovedFraudPolicy(
        model=model,
        metadata=metadata,
    )


def _validate_approval_contract(
    *,
    policy: dict[str, object],
    evaluation: dict[str, object],
    policy_path: Path,
    evaluation_path: Path,
) -> ApprovedPolicyMetadata:
    if policy.get("schema_version") != 1:
        raise ValueError("Threshold policy must use schema version 1.")

    if policy.get("selection_data") != "validation":
        raise ValueError("Threshold policy must be selected from validation data.")

    if policy.get("test_evaluation") != {"performed": False}:
        raise ValueError("Threshold policy must precede final test evaluation.")

    selected_policy = _require_object(
        policy,
        "selected_policy",
        description="Threshold policy",
    )
    model_name = _require_non_empty_string(
        selected_policy,
        "model_name",
        description="Selected policy",
    )
    model_artifact_name = _require_non_empty_string(
        selected_policy,
        "model_artifact",
        description="Selected policy",
    )

    if Path(model_artifact_name).name != model_artifact_name:
        raise ValueError("Selected model artifact must be a filename.")

    threshold = _require_probability(
        selected_policy,
        "threshold",
        description="Selected policy",
    )

    operating_constraint = _require_object(
        policy,
        "operating_constraint",
        description="Threshold policy",
    )
    daily_card_capacity = _require_positive_integer(
        operating_constraint,
        "daily_card_capacity",
        description="Operating constraint",
    )
    capacity_scope = _require_non_empty_string(
        operating_constraint,
        "scope",
        description="Operating constraint",
    )

    if capacity_scope != CAPACITY_SCOPE:
        raise ValueError(f"Operating constraint scope must be {CAPACITY_SCOPE}.")

    models = _require_object(
        policy,
        "models",
        description="Threshold policy",
    )
    selected_model = models.get(model_name)
    if not isinstance(selected_model, dict):
        raise ValueError("Threshold policy must contain metrics for the selected model.")

    if selected_model.get("model_artifact") != model_artifact_name:
        raise ValueError("Selected model artifact does not match its validation metrics.")

    validation_threshold = _require_probability(
        selected_model,
        "threshold",
        description="Selected model metrics",
    )
    _require_same_number(
        validation_threshold,
        threshold,
        message=("Selected threshold does not match its validation metrics."),
    )

    validation_transaction_metrics = _require_object(
        selected_model,
        "transaction_metrics",
        description="Selected model metrics",
    )
    validation_card_day_metrics = _require_object(
        selected_model,
        "card_day_metrics",
        description="Selected model metrics",
    )
    _validate_capacity_metrics(
        validation_card_day_metrics,
        threshold=threshold,
        daily_card_capacity=daily_card_capacity,
        description="Validation card-day metrics",
    )

    if evaluation.get("schema_version") != 1:
        raise ValueError("Final evaluation must use schema version 1.")

    if evaluation.get("evaluation_data") != "test":
        raise ValueError("Final evaluation must use the test partition.")

    if evaluation.get("selection_locked") is not True:
        raise ValueError("Final evaluation must use a locked policy.")

    if evaluation.get("policy_source") != policy_path.name:
        raise ValueError("Final evaluation does not reference the supplied policy.")

    if evaluation.get("model_name") != model_name:
        raise ValueError("Final evaluation model does not match the selected policy.")

    if evaluation.get("model_artifact") != model_artifact_name:
        raise ValueError("Final evaluation artifact does not match the selected policy.")

    evaluation_threshold = _require_probability(
        evaluation,
        "threshold",
        description="Final evaluation",
    )
    _require_same_number(
        evaluation_threshold,
        threshold,
        message=("Final evaluation threshold does not match the selected policy."),
    )

    evaluation_constraint = _require_object(
        evaluation,
        "operating_constraint",
        description="Final evaluation",
    )
    evaluation_capacity = _require_positive_integer(
        evaluation_constraint,
        "daily_card_capacity",
        description="Final evaluation operating constraint",
    )
    evaluation_scope = _require_non_empty_string(
        evaluation_constraint,
        "scope",
        description="Final evaluation operating constraint",
    )

    if evaluation_capacity != daily_card_capacity:
        raise ValueError("Final evaluation capacity does not match the selected policy.")

    if evaluation_scope != capacity_scope:
        raise ValueError("Final evaluation scope does not match the selected policy.")

    test_transaction_metrics = _require_object(
        evaluation,
        "transaction_metrics",
        description="Final evaluation",
    )
    test_card_day_metrics = _require_object(
        evaluation,
        "card_day_metrics",
        description="Final evaluation",
    )
    _validate_capacity_metrics(
        test_card_day_metrics,
        threshold=threshold,
        daily_card_capacity=daily_card_capacity,
        description="Test card-day metrics",
    )

    return ApprovedPolicyMetadata(
        model_name=model_name,
        model_artifact_name=model_artifact_name,
        threshold=threshold,
        daily_card_capacity=daily_card_capacity,
        capacity_scope=capacity_scope,
        policy_source=policy_path.name,
        evaluation_source=evaluation_path.name,
        validation_transaction_metrics=dict(validation_transaction_metrics),
        validation_card_day_metrics=dict(validation_card_day_metrics),
        test_transaction_metrics=dict(test_transaction_metrics),
        test_card_day_metrics=dict(test_card_day_metrics),
    )


def _validate_capacity_metrics(
    metrics: dict[str, object],
    *,
    threshold: float,
    daily_card_capacity: int,
    description: str,
) -> None:
    metric_threshold = _require_probability(
        metrics,
        "threshold",
        description=description,
    )
    _require_same_number(
        metric_threshold,
        threshold,
        message=f"{description} threshold does not match the policy.",
    )

    metric_capacity = _require_positive_integer(
        metrics,
        "daily_card_capacity",
        description=description,
    )
    if metric_capacity != daily_card_capacity:
        raise ValueError(f"{description} capacity does not match the policy.")

    budget_exceeded_days = _require_integer(
        metrics,
        "budget_exceeded_days",
        description=description,
    )
    if budget_exceeded_days != 0:
        raise ValueError(f"{description} must not exceed the daily capacity.")

    max_daily_card_alerts = _require_integer(
        metrics,
        "max_daily_card_alerts",
        description=description,
    )
    if max_daily_card_alerts < 0:
        raise ValueError(f"{description} maximum daily alerts must not be negative.")

    if max_daily_card_alerts > daily_card_capacity:
        raise ValueError(f"{description} maximum daily alerts exceed the policy capacity.")


def _load_json_object(
    path: Path,
    *,
    description: str,
) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{description} is not valid JSON: {path}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a JSON object.")

    return payload


def _require_object(
    payload: dict[str, object],
    key: str,
    *,
    description: str,
) -> dict[str, object]:
    value = payload.get(key)

    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain an object named {key}.")

    return value


def _require_non_empty_string(
    payload: dict[str, object],
    key: str,
    *,
    description: str,
) -> str:
    value = payload.get(key)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{description} must contain a non-empty {key}.")

    return value


def _require_probability(
    payload: dict[str, object],
    key: str,
    *,
    description: str,
) -> float:
    value = payload.get(key)

    if isinstance(value, bool):
        raise ValueError(f"{description} must contain a numeric {key}.")

    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{description} must contain a numeric {key}.") from exc

    if not math.isfinite(numeric_value):
        raise ValueError(f"{description} {key} must be finite.")

    if not 0.0 < numeric_value < 1.0:
        raise ValueError(f"{description} {key} must be between 0 and 1.")

    return numeric_value


def _require_positive_integer(
    payload: dict[str, object],
    key: str,
    *,
    description: str,
) -> int:
    value = _require_integer(
        payload,
        key,
        description=description,
    )

    if value <= 0:
        raise ValueError(f"{description} {key} must be greater than zero.")

    return value


def _require_integer(
    payload: dict[str, object],
    key: str,
    *,
    description: str,
) -> int:
    value = payload.get(key)

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{description} must contain an integer {key}.")

    return value


def _require_same_number(
    left: float,
    right: float,
    *,
    message: str,
) -> None:
    if not math.isclose(
        left,
        right,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError(message)
