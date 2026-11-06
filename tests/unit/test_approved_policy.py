import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from fraud_lakehouse.approved_policy import (
    ApprovedFraudPolicy,
    load_approved_fraud_policy,
)
from fraud_lakehouse.ml_dataset import (
    MODEL_FEATURE_COLUMNS,
    MODEL_TARGET_COLUMN,
)
from fraud_lakehouse.modeling import (
    build_model_preprocessor,
    fraud_probability,
)
from fraud_lakehouse.optimization import TunedModelArtifact


@dataclass(frozen=True)
class ApprovedPolicyFixture:
    model_dir: Path
    policy_path: Path
    evaluation_path: Path
    features: pd.DataFrame


@pytest.fixture
def approved_policy_fixture(
    tmp_path: Path,
) -> ApprovedPolicyFixture:
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    features = pd.DataFrame(
        {
            "tx_amount": [10.0, 250.0, 30.0, 400.0],
            "transaction_hour": [8, 2, 14, 1],
            "day_of_week": [1, 6, 2, 5],
            "is_weekend": [0, 1, 0, 1],
            "is_night": [0, 1, 0, 1],
            "customer_previous_transaction_count": [
                5,
                1,
                10,
                2,
            ],
            "customer_previous_mean_amount": [
                20.0,
                40.0,
                25.0,
                50.0,
            ],
            "amount_to_customer_previous_mean": [
                0.5,
                6.25,
                1.2,
                8.0,
            ],
            "terminal_previous_transaction_count": [
                100,
                5,
                200,
                10,
            ],
        }
    )
    target = pd.Series(
        [0, 1, 0, 1],
        name=MODEL_TARGET_COLUMN,
    )

    preprocessor = build_model_preprocessor()
    transformed_features = preprocessor.fit_transform(
        features,
        target,
    )
    estimator = LogisticRegression(
        random_state=42,
    )
    estimator.fit(
        transformed_features,
        target,
    )

    artifact = TunedModelArtifact(
        preprocessor=preprocessor,
        estimator=estimator,
        feature_columns=MODEL_FEATURE_COLUMNS,
        target_column=MODEL_TARGET_COLUMN,
        hyperparameters={},
    )
    joblib.dump(
        artifact,
        model_dir / "tuned_xgboost.joblib",
    )

    validation_card_day_metrics = {
        "threshold": 0.5,
        "daily_card_capacity": 100,
        "budget_exceeded_days": 0,
        "max_daily_card_alerts": 80,
    }
    test_card_day_metrics = {
        "threshold": 0.5,
        "daily_card_capacity": 100,
        "budget_exceeded_days": 0,
        "max_daily_card_alerts": 70,
    }

    policy = {
        "schema_version": 1,
        "selection_data": "validation",
        "test_evaluation": {
            "performed": False,
        },
        "operating_constraint": {
            "daily_card_capacity": 100,
            "scope": "unique_customer_cards_per_day",
        },
        "selected_policy": {
            "model_name": "xgboost",
            "model_artifact": "tuned_xgboost.joblib",
            "threshold": 0.5,
        },
        "models": {
            "xgboost": {
                "model_artifact": "tuned_xgboost.joblib",
                "threshold": 0.5,
                "transaction_metrics": {
                    "average_precision": 0.4,
                },
                "card_day_metrics": validation_card_day_metrics,
            }
        },
    }
    evaluation = {
        "schema_version": 1,
        "evaluation_data": "test",
        "selection_locked": True,
        "policy_source": "threshold_policy.json",
        "model_name": "xgboost",
        "model_artifact": "tuned_xgboost.joblib",
        "threshold": 0.5,
        "operating_constraint": {
            "daily_card_capacity": 100,
            "scope": "unique_customer_cards_per_day",
        },
        "transaction_metrics": {
            "average_precision": 0.35,
        },
        "card_day_metrics": test_card_day_metrics,
    }

    policy_path = model_dir / "threshold_policy.json"
    evaluation_path = model_dir / "final_evaluation.json"

    _write_json(
        policy_path,
        policy,
    )
    _write_json(
        evaluation_path,
        evaluation,
    )

    return ApprovedPolicyFixture(
        model_dir=model_dir,
        policy_path=policy_path,
        evaluation_path=evaluation_path,
        features=features,
    )


def test_loads_and_scores_approved_policy(
    approved_policy_fixture: ApprovedPolicyFixture,
) -> None:
    approved_policy = _load_fixture_policy(approved_policy_fixture)

    predictions = approved_policy.predict(
        approved_policy_fixture.features.assign(ignored_metadata="preserved-outside-model-contract")
    )

    transformed_features = approved_policy.model.preprocessor.transform(
        approved_policy_fixture.features
    )
    expected_probabilities = fraud_probability(
        approved_policy.model.estimator,
        transformed_features,
    )

    assert approved_policy.metadata.model_name == "xgboost"
    assert approved_policy.metadata.model_artifact_name == "tuned_xgboost.joblib"
    assert approved_policy.metadata.threshold == 0.5
    assert approved_policy.metadata.daily_card_capacity == 100
    assert list(predictions.columns) == [
        "fraud_probability",
        "threshold_decision",
    ]
    assert predictions.index.equals(approved_policy_fixture.features.index)
    assert predictions["threshold_decision"].dtype == bool

    np.testing.assert_allclose(
        predictions["fraud_probability"],
        expected_probabilities,
    )
    assert predictions["threshold_decision"].tolist() == (expected_probabilities >= 0.5).tolist()


def test_approved_policy_survives_joblib_round_trip(
    approved_policy_fixture: ApprovedPolicyFixture,
) -> None:
    approved_policy = _load_fixture_policy(approved_policy_fixture)
    artifact_path = approved_policy_fixture.model_dir / "approved_fraud_policy.joblib"

    expected_predictions = approved_policy.predict(approved_policy_fixture.features)

    joblib.dump(
        approved_policy,
        artifact_path,
    )
    restored_policy = joblib.load(artifact_path)

    assert isinstance(restored_policy, ApprovedFraudPolicy)
    assert restored_policy.metadata == approved_policy.metadata

    actual_predictions = restored_policy.predict(approved_policy_fixture.features)
    pd.testing.assert_frame_equal(
        actual_predictions,
        expected_predictions,
    )


def test_rejects_missing_model_features(
    approved_policy_fixture: ApprovedPolicyFixture,
) -> None:
    approved_policy = _load_fixture_policy(approved_policy_fixture)
    incomplete_features = approved_policy_fixture.features.drop(columns="tx_amount")

    with pytest.raises(
        ValueError,
        match="Missing required model features: tx_amount",
    ):
        approved_policy.predict(incomplete_features)


def test_rejects_evaluation_threshold_mismatch(
    approved_policy_fixture: ApprovedPolicyFixture,
) -> None:
    evaluation = _read_json(approved_policy_fixture.evaluation_path)
    evaluation["threshold"] = 0.6
    _write_json(
        approved_policy_fixture.evaluation_path,
        evaluation,
    )

    with pytest.raises(
        ValueError,
        match=("Final evaluation threshold does not match the selected policy"),
    ):
        _load_fixture_policy(approved_policy_fixture)


def test_rejects_test_capacity_violation(
    approved_policy_fixture: ApprovedPolicyFixture,
) -> None:
    evaluation = _read_json(approved_policy_fixture.evaluation_path)
    card_day_metrics = evaluation["card_day_metrics"]
    assert isinstance(card_day_metrics, dict)
    card_day_metrics["budget_exceeded_days"] = 1

    _write_json(
        approved_policy_fixture.evaluation_path,
        evaluation,
    )

    with pytest.raises(
        ValueError,
        match=("Test card-day metrics must not exceed the daily capacity"),
    ):
        _load_fixture_policy(approved_policy_fixture)


def test_rejects_unsafe_model_artifact_path(
    approved_policy_fixture: ApprovedPolicyFixture,
) -> None:
    policy = _read_json(approved_policy_fixture.policy_path)
    selected_policy = policy["selected_policy"]
    assert isinstance(selected_policy, dict)
    selected_policy["model_artifact"] = "../tuned_xgboost.joblib"

    _write_json(
        approved_policy_fixture.policy_path,
        policy,
    )

    with pytest.raises(
        ValueError,
        match="Selected model artifact must be a filename",
    ):
        _load_fixture_policy(approved_policy_fixture)


def _load_fixture_policy(
    fixture: ApprovedPolicyFixture,
) -> ApprovedFraudPolicy:
    return load_approved_fraud_policy(
        model_dir=fixture.model_dir,
        policy_path=fixture.policy_path,
        evaluation_path=fixture.evaluation_path,
    )


def _read_json(
    path: Path,
) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_json(
    path: Path,
    payload: dict[str, object],
) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
