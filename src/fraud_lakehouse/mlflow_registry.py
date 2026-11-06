import math
from dataclasses import dataclass
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModel, PythonModelContext

from fraud_lakehouse.approved_policy import (
    ApprovedFraudPolicy,
    load_approved_fraud_policy,
)

DEFAULT_MLFLOW_EXPERIMENT = "fraud-policy-registration"
DEFAULT_REGISTERED_MODEL_NAME = "fraud-detection-policy"
DEFAULT_MODEL_ALIAS = "champion"
MLFLOW_MODEL_ARTIFACT_NAME = "fraud_policy"

MODEL_PIP_REQUIREMENTS = (
    "mlflow-skinny==3.10.1",
    "pandas>=2.2",
    "scikit-learn>=1.9,<2.0",
    "xgboost>=3.2,<3.3",
)


@dataclass(frozen=True)
class RegisteredPolicyVersion:
    """Identity of a validated policy version published to MLflow."""

    run_id: str
    logged_model_uri: str
    registered_model_name: str
    registered_model_version: str
    registered_model_uri: str
    alias: str


class FraudPolicyPythonModel(PythonModel):
    """MLflow PyFunc adapter for the approved fraud policy."""

    def __init__(
        self,
        policy: ApprovedFraudPolicy,
    ) -> None:
        self.policy = policy

    def predict(
        self,
        context: PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, object] | None = None,
    ) -> pd.DataFrame:
        """Return fraud probabilities and locked-threshold decisions."""
        return self.policy.predict(model_input)


def publish_approved_fraud_policy(
    *,
    tracking_uri: str,
    experiment_name: str,
    registered_model_name: str,
    alias: str,
    model_dir: Path,
    policy_path: Path,
    evaluation_path: Path,
    input_example: pd.DataFrame,
    source_revision: str,
) -> RegisteredPolicyVersion:
    """Log, verify, register and promote an approved fraud policy."""
    validated_tracking_uri = _require_text(
        tracking_uri,
        name="tracking_uri",
    )
    validated_experiment_name = _require_text(
        experiment_name,
        name="experiment_name",
    )
    validated_registered_model_name = _require_text(
        registered_model_name,
        name="registered_model_name",
    )
    validated_alias = _require_text(
        alias,
        name="alias",
    )
    validated_source_revision = _require_text(
        source_revision,
        name="source_revision",
    )

    policy = load_approved_fraud_policy(
        model_dir=Path(model_dir),
        policy_path=Path(policy_path),
        evaluation_path=Path(evaluation_path),
    )
    model_input = _prepare_input_example(
        input_example,
        policy=policy,
    )
    expected_predictions = policy.predict(model_input)
    signature = infer_signature(
        model_input,
        expected_predictions,
    )

    mlflow.set_tracking_uri(validated_tracking_uri)
    experiment = mlflow.set_experiment(validated_experiment_name)

    run_tags = {
        "approval_status": "approved",
        "model_name": policy.metadata.model_name,
        "source_revision": validated_source_revision,
    }

    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name=(f"register-{validated_registered_model_name}"),
        tags=run_tags,
    ) as run:
        mlflow.log_params(
            {
                "selected_model": policy.metadata.model_name,
                "selected_model_artifact": (policy.metadata.model_artifact_name),
                "decision_threshold": policy.metadata.threshold,
                "daily_card_capacity": (policy.metadata.daily_card_capacity),
                "capacity_scope": (policy.metadata.capacity_scope),
                "feature_count": len(policy.model.feature_columns),
                "target_column": policy.model.target_column,
            }
        )
        mlflow.log_metrics(_approved_policy_metrics(policy))
        mlflow.log_artifact(
            str(policy_path),
            artifact_path="approval",
        )
        mlflow.log_artifact(
            str(evaluation_path),
            artifact_path="approval",
        )

        model_info = mlflow.pyfunc.log_model(
            name=MLFLOW_MODEL_ARTIFACT_NAME,
            python_model=FraudPolicyPythonModel(policy),
            signature=signature,
            input_example=model_input,
            pip_requirements=list(MODEL_PIP_REQUIREMENTS),
            metadata={
                "approval_status": "approved",
                "model_name": policy.metadata.model_name,
                "threshold": policy.metadata.threshold,
                "daily_card_capacity": (policy.metadata.daily_card_capacity),
                "capacity_scope": (policy.metadata.capacity_scope),
                "source_revision": validated_source_revision,
            },
        )

        _verify_model_predictions(
            model_uri=model_info.model_uri,
            model_input=model_input,
            expected_predictions=expected_predictions,
            description="Logged model",
        )

        registered_version = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=validated_registered_model_name,
            await_registration_for=300,
        )
        registered_model_version = str(registered_version.version)
        registered_model_uri = (
            f"models:/{validated_registered_model_name}/{registered_model_version}"
        )

        _verify_model_predictions(
            model_uri=registered_model_uri,
            model_input=model_input,
            expected_predictions=expected_predictions,
            description="Registered model",
        )

        client = MlflowClient(tracking_uri=validated_tracking_uri)
        version_tags = {
            "approval_status": "approved",
            "model_name": policy.metadata.model_name,
            "source_revision": validated_source_revision,
            "threshold": str(policy.metadata.threshold),
            "daily_card_capacity": str(policy.metadata.daily_card_capacity),
        }

        for key, value in version_tags.items():
            client.set_model_version_tag(
                name=validated_registered_model_name,
                version=registered_model_version,
                key=key,
                value=value,
            )

        client.set_registered_model_alias(
            name=validated_registered_model_name,
            alias=validated_alias,
            version=registered_model_version,
        )

        return RegisteredPolicyVersion(
            run_id=run.info.run_id,
            logged_model_uri=model_info.model_uri,
            registered_model_name=(validated_registered_model_name),
            registered_model_version=(registered_model_version),
            registered_model_uri=registered_model_uri,
            alias=validated_alias,
        )


def _prepare_input_example(
    input_example: pd.DataFrame,
    *,
    policy: ApprovedFraudPolicy,
) -> pd.DataFrame:
    if not isinstance(input_example, pd.DataFrame):
        raise TypeError("input_example must be a pandas DataFrame.")

    if input_example.empty:
        raise ValueError("input_example must contain at least one row.")

    missing_columns = sorted(set(policy.model.feature_columns).difference(input_example.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Input example is missing model features: {missing}")

    return input_example.loc[
        :,
        list(policy.model.feature_columns),
    ].copy()


def _verify_model_predictions(
    *,
    model_uri: str,
    model_input: pd.DataFrame,
    expected_predictions: pd.DataFrame,
    description: str,
) -> None:
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    actual_predictions = loaded_model.predict(model_input)

    if not isinstance(actual_predictions, pd.DataFrame):
        raise TypeError(f"{description} must return a pandas DataFrame.")

    try:
        pd.testing.assert_frame_equal(
            actual_predictions.reset_index(drop=True),
            expected_predictions.reset_index(drop=True),
        )
    except AssertionError as exc:
        raise ValueError(
            f"{description} predictions do not match the approved source policy."
        ) from exc


def _approved_policy_metrics(
    policy: ApprovedFraudPolicy,
) -> dict[str, float]:
    metric_groups = {
        "validation_transaction": (policy.metadata.validation_transaction_metrics),
        "validation_card_day": (policy.metadata.validation_card_day_metrics),
        "test_transaction": (policy.metadata.test_transaction_metrics),
        "test_card_day": (policy.metadata.test_card_day_metrics),
    }

    metrics: dict[str, float] = {}

    for prefix, values in metric_groups.items():
        for key, value in values.items():
            if isinstance(value, bool):
                continue

            if not isinstance(value, (int, float)):
                continue

            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise ValueError(f"Approved policy metric is not finite: {prefix}_{key}")

            metrics[f"{prefix}_{key}"] = numeric_value

    return metrics


def _require_text(
    value: str,
    *,
    name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be empty.")

    return value.strip()
