from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import fraud_lakehouse.mlflow_registry as registry
from fraud_lakehouse.mlflow_registry import (
    FraudPolicyPythonModel,
    RegisteredPolicyVersion,
)


class FakeApprovedPolicy:
    """Small approved-policy substitute for registry unit tests."""

    def __init__(self) -> None:
        self.model = SimpleNamespace(
            feature_columns=("feature_a", "feature_b"),
            target_column="tx_fraud",
        )
        self.metadata = SimpleNamespace(
            model_name="xgboost",
            model_artifact_name="tuned_xgboost.joblib",
            threshold=0.6,
            daily_card_capacity=100,
            capacity_scope="unique_customer_cards_per_day",
            validation_transaction_metrics={
                "average_precision": 0.40,
                "true_positives": 10,
            },
            validation_card_day_metrics={
                "card_day_recall": 0.30,
                "budget_exceeded_days": 0,
            },
            test_transaction_metrics={
                "average_precision": 0.35,
                "selection_locked": True,
            },
            test_card_day_metrics={
                "card_day_recall": 0.28,
                "description": "ignored",
            },
        )

    def predict(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        probabilities = (features["feature_a"] + features["feature_b"]) / 10.0

        return pd.DataFrame(
            {
                "fraud_probability": probabilities,
                "threshold_decision": probabilities >= 0.6,
            },
            index=features.index,
        )


def test_pyfunc_adapter_delegates_to_approved_policy() -> None:
    policy = FakeApprovedPolicy()
    model = FraudPolicyPythonModel(policy)
    model_input = pd.DataFrame(
        {
            "feature_a": [1.0, 4.0],
            "feature_b": [2.0, 3.0],
        }
    )

    predictions = model.predict(
        context=None,
        model_input=model_input,
    )

    expected = policy.predict(model_input)
    pd.testing.assert_frame_equal(
        predictions,
        expected,
    )


def test_prepare_input_example_selects_feature_contract() -> None:
    policy = FakeApprovedPolicy()
    input_example = pd.DataFrame(
        {
            "ignored": ["metadata"],
            "feature_b": [2.0],
            "feature_a": [1.0],
        }
    )

    prepared = registry._prepare_input_example(
        input_example,
        policy=policy,
    )

    assert list(prepared.columns) == [
        "feature_a",
        "feature_b",
    ]
    assert prepared.to_dict(orient="records") == [
        {
            "feature_a": 1.0,
            "feature_b": 2.0,
        }
    ]


def test_prepare_input_example_rejects_missing_features() -> None:
    policy = FakeApprovedPolicy()
    input_example = pd.DataFrame(
        {
            "feature_a": [1.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="Input example is missing model features: feature_b",
    ):
        registry._prepare_input_example(
            input_example,
            policy=policy,
        )


def test_approved_policy_metrics_flattens_numeric_values() -> None:
    policy = FakeApprovedPolicy()

    metrics = registry._approved_policy_metrics(policy)

    assert metrics == {
        "validation_transaction_average_precision": 0.40,
        "validation_transaction_true_positives": 10.0,
        "validation_card_day_card_day_recall": 0.30,
        "validation_card_day_budget_exceeded_days": 0.0,
        "test_transaction_average_precision": 0.35,
        "test_card_day_card_day_recall": 0.28,
    }


def test_approved_policy_metrics_rejects_non_finite_values() -> None:
    policy = FakeApprovedPolicy()
    policy.metadata.test_transaction_metrics = {
        "average_precision": float("nan"),
    }

    with pytest.raises(
        ValueError,
        match=("Approved policy metric is not finite: test_transaction_average_precision"),
    ):
        registry._approved_policy_metrics(policy)


def test_verify_model_predictions_rejects_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_input = pd.DataFrame(
        {
            "feature_a": [1.0],
            "feature_b": [2.0],
        }
    )
    expected_predictions = pd.DataFrame(
        {
            "fraud_probability": [0.3],
            "threshold_decision": [False],
        }
    )
    drifted_predictions = pd.DataFrame(
        {
            "fraud_probability": [0.9],
            "threshold_decision": [True],
        }
    )
    loaded_model = SimpleNamespace(predict=lambda _: drifted_predictions)

    monkeypatch.setattr(
        registry.mlflow.pyfunc,
        "load_model",
        lambda _: loaded_model,
    )

    with pytest.raises(
        ValueError,
        match=("Registered model predictions do not match the approved source policy"),
    ):
        registry._verify_model_predictions(
            model_uri="models:/fraud-detection-policy/1",
            model_input=model_input,
            expected_predictions=expected_predictions,
            description="Registered model",
        )


def test_publisher_promotes_alias_after_both_verifications(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = FakeApprovedPolicy()
    input_example = pd.DataFrame(
        {
            "feature_a": [1.0],
            "feature_b": [2.0],
        }
    )
    events: list[str] = []

    monkeypatch.setattr(
        registry,
        "load_approved_fraud_policy",
        lambda **_: policy,
    )
    monkeypatch.setattr(
        registry,
        "infer_signature",
        lambda *_: "signature",
    )
    monkeypatch.setattr(
        registry.mlflow,
        "set_tracking_uri",
        lambda _: None,
    )
    monkeypatch.setattr(
        registry.mlflow,
        "set_experiment",
        lambda _: SimpleNamespace(experiment_id="experiment-1"),
    )

    @contextmanager
    def fake_start_run(
        **_: Any,
    ) -> Any:
        yield SimpleNamespace(info=SimpleNamespace(run_id="run-1"))

    monkeypatch.setattr(
        registry.mlflow,
        "start_run",
        fake_start_run,
    )
    monkeypatch.setattr(
        registry.mlflow,
        "log_params",
        lambda _: None,
    )
    monkeypatch.setattr(
        registry.mlflow,
        "log_metrics",
        lambda _: None,
    )
    monkeypatch.setattr(
        registry.mlflow,
        "log_artifact",
        lambda *_, **__: None,
    )

    def fake_log_model(
        **_: Any,
    ) -> Any:
        events.append("log")
        return SimpleNamespace(model_uri="models:/logged-model-id")

    monkeypatch.setattr(
        registry.mlflow.pyfunc,
        "log_model",
        fake_log_model,
    )

    def fake_verify(
        *,
        description: str,
        **_: Any,
    ) -> None:
        events.append(f"verify:{description}")

    monkeypatch.setattr(
        registry,
        "_verify_model_predictions",
        fake_verify,
    )

    def fake_register_model(
        **_: Any,
    ) -> Any:
        events.append("register")
        return SimpleNamespace(version="7")

    monkeypatch.setattr(
        registry.mlflow,
        "register_model",
        fake_register_model,
    )

    class FakeClient:
        def set_model_version_tag(
            self,
            **_: Any,
        ) -> None:
            events.append("tag")

        def set_registered_model_alias(
            self,
            **_: Any,
        ) -> None:
            events.append("alias")

    monkeypatch.setattr(
        registry,
        "MlflowClient",
        lambda **_: FakeClient(),
    )

    result = registry.publish_approved_fraud_policy(
        tracking_uri="http://127.0.0.1:5000",
        experiment_name="fraud-policy-registration",
        registered_model_name="fraud-detection-policy",
        alias="champion",
        model_dir=Path("data/models"),
        policy_path=Path("data/models/threshold_policy.json"),
        evaluation_path=Path("data/models/final_evaluation.json"),
        input_example=input_example,
        source_revision="abc123",
    )

    assert isinstance(result, RegisteredPolicyVersion)
    assert result.run_id == "run-1"
    assert result.registered_model_version == "7"
    assert result.registered_model_uri == ("models:/fraud-detection-policy/7")
    assert result.alias == "champion"

    assert events.index("log") < events.index("verify:Logged model")
    assert events.index("verify:Logged model") < events.index("register")
    assert events.index("register") < events.index("verify:Registered model")
    assert events.index("verify:Registered model") < events.index("alias")
