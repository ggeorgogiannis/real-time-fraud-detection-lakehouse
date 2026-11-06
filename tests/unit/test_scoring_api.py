from collections.abc import Iterator
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import fraud_lakehouse.scoring_api as scoring_api
from fraud_lakehouse.ml_dataset import MODEL_FEATURE_COLUMNS
from fraud_lakehouse.scoring_api import (
    RegisteredModelMetadata,
    ScoringSettings,
    create_app,
    resolve_registered_model_metadata,
)


class DeterministicScoringModel:
    """Predictable model substitute used by API tests."""

    def __init__(self) -> None:
        self.last_input: pd.DataFrame | None = None

    def predict(
        self,
        model_input: pd.DataFrame,
    ) -> pd.DataFrame:
        self.last_input = model_input.copy()

        probability = (model_input["tx_amount"].astype(float) / 1000.0).clip(
            lower=0.0,
            upper=1.0,
        )

        return pd.DataFrame(
            {
                "fraud_probability": probability,
                "threshold_decision": probability >= 0.6,
            },
            index=model_input.index,
        )


@pytest.fixture
def scoring_client() -> Iterator[tuple[TestClient, DeterministicScoringModel]]:
    model = DeterministicScoringModel()
    application = create_app(
        settings=_settings(),
        model_loader=lambda _: model,
        metadata_resolver=lambda _: _metadata(),
    )

    with TestClient(application) as client:
        yield client, model


def test_health_endpoints_report_loaded_model(
    scoring_client: tuple[
        TestClient,
        DeterministicScoringModel,
    ],
) -> None:
    client, _ = scoring_client

    live_response = client.get("/health/live")
    ready_response = client.get("/health/ready")

    assert live_response.status_code == 200
    assert live_response.json() == {
        "status": "alive",
    }

    assert ready_response.status_code == 200
    assert ready_response.json() == {
        "status": "ready",
        "model_name": "fraud-detection-policy",
        "model_version": "7",
        "model_alias": "champion",
    }


def test_scores_transactions_and_preserves_identity(
    scoring_client: tuple[
        TestClient,
        DeterministicScoringModel,
    ],
) -> None:
    client, _ = scoring_client

    response = client.post(
        "/v1/score",
        json={
            "transactions": [
                _transaction(
                    transaction_id=1001,
                    tx_amount=100.0,
                ),
                _transaction(
                    transaction_id=1002,
                    tx_amount=800.0,
                ),
            ]
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "model": {
            "name": "fraud-detection-policy",
            "version": "7",
            "alias": "champion",
            "source_revision": "abc123",
            "threshold": 0.6,
            "daily_card_capacity": 100,
            "capacity_scope": ("unique_customer_cards_per_day"),
            "capacity_enforced_by_endpoint": False,
        },
        "predictions": [
            {
                "transaction_id": 1001,
                "customer_id": 101,
                "terminal_id": 201,
                "fraud_probability": 0.1,
                "threshold_decision": False,
            },
            {
                "transaction_id": 1002,
                "customer_id": 101,
                "terminal_id": 201,
                "fraud_probability": 0.8,
                "threshold_decision": True,
            },
        ],
    }


def test_passes_only_ordered_model_features(
    scoring_client: tuple[
        TestClient,
        DeterministicScoringModel,
    ],
) -> None:
    client, model = scoring_client
    transaction = _transaction(
        transaction_id=1001,
        tx_amount=100.0,
    )
    transaction["customer_previous_mean_amount"] = None
    transaction["amount_to_customer_previous_mean"] = None

    response = client.post(
        "/v1/score",
        json={
            "transactions": [transaction],
        },
    )

    assert response.status_code == 200
    assert model.last_input is not None
    assert list(model.last_input.columns) == list(MODEL_FEATURE_COLUMNS)
    assert model.last_input.dtypes.astype(str).to_dict() == {
        "tx_amount": "float64",
        "transaction_hour": "int32",
        "day_of_week": "int32",
        "is_weekend": "int32",
        "is_night": "int32",
        "customer_previous_transaction_count": "int64",
        "customer_previous_mean_amount": "float64",
        "amount_to_customer_previous_mean": "float64",
        "terminal_previous_transaction_count": "int64",
    }
    assert "transaction_id" not in model.last_input.columns
    assert "customer_id" not in model.last_input.columns
    assert "terminal_id" not in model.last_input.columns
    assert "tx_datetime" not in model.last_input.columns
    assert pd.isna(
        model.last_input.loc[
            0,
            "customer_previous_mean_amount",
        ]
    )
    assert pd.isna(
        model.last_input.loc[
            0,
            "amount_to_customer_previous_mean",
        ]
    )


def test_rejects_batch_larger_than_configured_limit() -> None:
    model = DeterministicScoringModel()
    application = create_app(
        settings=_settings(max_batch_size=2),
        model_loader=lambda _: model,
        metadata_resolver=lambda _: _metadata(),
    )

    with TestClient(application) as client:
        response = client.post(
            "/v1/score",
            json={
                "transactions": [
                    _transaction(
                        transaction_id=1001,
                        tx_amount=100.0,
                    ),
                    _transaction(
                        transaction_id=1002,
                        tx_amount=200.0,
                    ),
                    _transaction(
                        transaction_id=1003,
                        tx_amount=300.0,
                    ),
                ]
            },
        )

    assert response.status_code == 413
    assert response.json() == {"detail": ("Request exceeds the maximum batch size of 2")}


def test_rejects_duplicate_transaction_ids(
    scoring_client: tuple[
        TestClient,
        DeterministicScoringModel,
    ],
) -> None:
    client, _ = scoring_client

    response = client.post(
        "/v1/score",
        json={
            "transactions": [
                _transaction(
                    transaction_id=1001,
                    tx_amount=100.0,
                ),
                _transaction(
                    transaction_id=1001,
                    tx_amount=200.0,
                ),
            ]
        },
    )

    assert response.status_code == 422
    assert "transaction_id values must be unique" in response.text


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("transaction_hour", 24),
        ("day_of_week", 7),
        ("is_weekend", 2),
        ("tx_amount", -1.0),
        ("customer_previous_transaction_count", -1),
        ("unexpected_feature", 1),
    ],
)
def test_rejects_invalid_transaction_features(
    scoring_client: tuple[
        TestClient,
        DeterministicScoringModel,
    ],
    field: str,
    value: object,
) -> None:
    client, _ = scoring_client
    transaction = _transaction(
        transaction_id=1001,
        tx_amount=100.0,
    )
    transaction[field] = value

    response = client.post(
        "/v1/score",
        json={
            "transactions": [transaction],
        },
    )

    assert response.status_code == 422


def test_rejects_timestamp_without_timezone(
    scoring_client: tuple[
        TestClient,
        DeterministicScoringModel,
    ],
) -> None:
    client, _ = scoring_client
    transaction = _transaction(
        transaction_id=1001,
        tx_amount=100.0,
    )
    transaction["tx_datetime"] = "2018-04-01T08:00:00"

    response = client.post(
        "/v1/score",
        json={
            "transactions": [transaction],
        },
    )

    assert response.status_code == 422
    assert "tx_datetime must include a timezone" in response.text


def test_returns_internal_error_for_invalid_model_output() -> None:
    invalid_model = SimpleNamespace(
        predict=lambda _: pd.DataFrame(
            {
                "fraud_probability": [1.5],
                "threshold_decision": [True],
            }
        )
    )
    application = create_app(
        settings=_settings(),
        model_loader=lambda _: invalid_model,
        metadata_resolver=lambda _: _metadata(),
    )

    with TestClient(application) as client:
        response = client.post(
            "/v1/score",
            json={
                "transactions": [
                    _transaction(
                        transaction_id=1001,
                        tx_amount=100.0,
                    )
                ]
            },
        )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Model prediction failed",
    }


def test_exposes_scoring_metrics(
    scoring_client: tuple[
        TestClient,
        DeterministicScoringModel,
    ],
) -> None:
    client, _ = scoring_client

    score_response = client.post(
        "/v1/score",
        json={
            "transactions": [
                _transaction(
                    transaction_id=1001,
                    tx_amount=100.0,
                ),
                _transaction(
                    transaction_id=1002,
                    tx_amount=800.0,
                ),
            ]
        },
    )
    metrics_response = client.get("/metrics")

    assert score_response.status_code == 200
    assert metrics_response.status_code == 200
    assert "text/plain" in metrics_response.headers["content-type"]

    metrics = metrics_response.text

    assert "fraud_scoring_transactions_total 2.0" in metrics
    assert "fraud_scoring_threshold_decisions_total 1.0" in metrics
    assert (
        "fraud_scoring_requests_total{"
        'endpoint="/v1/score",'
        'method="POST",'
        'status_code="200"} 1.0' in metrics
    )
    assert (
        "fraud_scoring_model_info{"
        'model_alias="champion",'
        'model_name="fraud-detection-policy",'
        'model_version="7",'
        'source_revision="abc123"} 1.0' in metrics
    )


def test_resolves_pinned_approved_model_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_version = SimpleNamespace(
        version="7",
        tags={
            "approval_status": "approved",
            "source_revision": "abc123",
            "threshold": "0.6",
            "daily_card_capacity": "100",
        },
    )
    client = SimpleNamespace(get_model_version_by_alias=lambda *_: model_version)
    monkeypatch.setattr(
        scoring_api,
        "MlflowClient",
        lambda **_: client,
    )

    metadata = resolve_registered_model_metadata(_settings())

    assert metadata == _metadata()


def test_rejects_unapproved_model_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_version = SimpleNamespace(
        version="7",
        tags={
            "approval_status": "candidate",
            "source_revision": "abc123",
            "threshold": "0.6",
            "daily_card_capacity": "100",
        },
    )
    client = SimpleNamespace(get_model_version_by_alias=lambda *_: model_version)
    monkeypatch.setattr(
        scoring_api,
        "MlflowClient",
        lambda **_: client,
    )

    with pytest.raises(
        ValueError,
        match=("Registered model alias must reference an approved model version"),
    ):
        resolve_registered_model_metadata(_settings())


def _settings(
    *,
    max_batch_size: int = 100,
) -> ScoringSettings:
    return ScoringSettings(
        tracking_uri="http://mlflow:5000",
        registered_model_name="fraud-detection-policy",
        model_alias="champion",
        max_batch_size=max_batch_size,
    )


def _metadata() -> RegisteredModelMetadata:
    return RegisteredModelMetadata(
        model_name="fraud-detection-policy",
        model_version="7",
        model_alias="champion",
        model_uri="models:/fraud-detection-policy/7",
        source_revision="abc123",
        threshold=0.6,
        daily_card_capacity=100,
        capacity_scope="unique_customer_cards_per_day",
    )


def _transaction(
    *,
    transaction_id: int,
    tx_amount: float,
) -> dict[str, object]:
    return {
        "transaction_id": transaction_id,
        "tx_datetime": "2018-04-01T08:00:00Z",
        "customer_id": 101,
        "terminal_id": 201,
        "tx_amount": tx_amount,
        "transaction_hour": 8,
        "day_of_week": 6,
        "is_weekend": 1,
        "is_night": 0,
        "customer_previous_transaction_count": 5,
        "customer_previous_mean_amount": 20.0,
        "amount_to_customer_previous_mean": 5.0,
        "terminal_previous_transaction_count": 100,
    }
