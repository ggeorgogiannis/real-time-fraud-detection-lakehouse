import numpy as np
import pandas as pd
import pytest

from fraud_lakehouse.thresholds import (
    CapacityConstrainedThreshold,
    ModelThresholdCandidate,
    select_capacity_constrained_threshold,
    select_model_threshold_policy,
)


def test_select_threshold_maximizes_card_recall_within_daily_capacity() -> None:
    partition = pd.DataFrame(
        {
            "transaction_date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-01",
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-02",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-03",
                    "2026-01-03",
                ]
            ).date,
            "customer_id": range(1, 10),
            "tx_fraud": [
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                1,
            ],
        }
    )
    fraud_probability = [
        0.90,
        0.80,
        0.70,
        0.95,
        0.85,
        0.75,
        0.90,
        0.80,
        0.74,
    ]

    result = select_capacity_constrained_threshold(
        partition,
        fraud_probability,
        daily_card_capacity=2,
    )

    assert result.threshold == pytest.approx(np.nextafter(0.75, np.inf))
    assert result.card_alert_count == 6
    assert result.mean_daily_card_alerts == pytest.approx(2.0)
    assert result.max_daily_card_alerts == 2
    assert result.budget_exceeded_days == 0

    assert result.true_positive_card_days == 2
    assert result.false_positive_card_days == 4
    assert result.false_negative_card_days == 1
    assert result.true_negative_card_days == 2

    assert result.card_day_precision == pytest.approx(1 / 3)
    assert result.card_day_recall == pytest.approx(2 / 3)
    assert result.card_day_f1 == pytest.approx(4 / 9)


def test_select_threshold_aggregates_transactions_by_card_day() -> None:
    partition = pd.DataFrame(
        {
            "transaction_date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-01",
                    "2026-01-01",
                ]
            ).date,
            "customer_id": [1, 1, 2],
            "tx_fraud": [0, 1, 0],
        }
    )

    result = select_capacity_constrained_threshold(
        partition,
        [0.90, 0.40, 0.80],
        daily_card_capacity=1,
    )

    assert result.threshold == pytest.approx(np.nextafter(0.80, np.inf))
    assert result.card_alert_count == 1
    assert result.max_daily_card_alerts == 1
    assert result.budget_exceeded_days == 0

    assert result.true_positive_card_days == 1
    assert result.false_positive_card_days == 0
    assert result.false_negative_card_days == 0
    assert result.true_negative_card_days == 1
    assert result.card_day_precision == pytest.approx(1.0)
    assert result.card_day_recall == pytest.approx(1.0)


@pytest.fixture
def valid_threshold_partition() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_date": pd.to_datetime(["2026-01-01", "2026-01-01"]).date,
            "customer_id": [1, 2],
            "tx_fraud": [1, 0],
        }
    )


@pytest.mark.parametrize("daily_card_capacity", [0, -1])
def test_select_threshold_rejects_non_positive_capacity(
    valid_threshold_partition: pd.DataFrame,
    daily_card_capacity: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="daily_card_capacity must be greater than zero",
    ):
        select_capacity_constrained_threshold(
            valid_threshold_partition,
            [0.90, 0.80],
            daily_card_capacity=daily_card_capacity,
        )


def test_select_threshold_rejects_missing_required_columns(
    valid_threshold_partition: pd.DataFrame,
) -> None:
    partition = valid_threshold_partition.drop(columns="customer_id")

    with pytest.raises(
        ValueError,
        match="Missing required columns: customer_id",
    ):
        select_capacity_constrained_threshold(
            partition,
            [0.90, 0.80],
            daily_card_capacity=1,
        )


def test_select_threshold_rejects_probability_length_mismatch(
    valid_threshold_partition: pd.DataFrame,
) -> None:
    with pytest.raises(
        ValueError,
        match="must have equal lengths",
    ):
        select_capacity_constrained_threshold(
            valid_threshold_partition,
            [0.90],
            daily_card_capacity=1,
        )


@pytest.mark.parametrize(
    ("fraud_probability", "message"),
    [
        ([np.nan, 0.80], "must contain finite values"),
        ([1.10, 0.80], "must contain values between 0 and 1"),
        ([-0.10, 0.80], "must contain values between 0 and 1"),
    ],
)
def test_select_threshold_rejects_invalid_probabilities(
    valid_threshold_partition: pd.DataFrame,
    fraud_probability: list[float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        select_capacity_constrained_threshold(
            valid_threshold_partition,
            fraud_probability,
            daily_card_capacity=1,
        )


def test_select_model_policy_prioritizes_card_day_recall() -> None:
    logistic_metrics = CapacityConstrainedThreshold(
        threshold=0.80,
        daily_card_capacity=100,
        card_alert_count=200,
        mean_daily_card_alerts=50.0,
        max_daily_card_alerts=100,
        budget_exceeded_days=0,
        card_day_precision=0.60,
        card_day_recall=0.50,
        card_day_f1=0.545,
        true_negative_card_days=800,
        false_positive_card_days=80,
        false_negative_card_days=50,
        true_positive_card_days=50,
    )
    xgboost_metrics = CapacityConstrainedThreshold(
        threshold=0.90,
        daily_card_capacity=100,
        card_alert_count=220,
        mean_daily_card_alerts=55.0,
        max_daily_card_alerts=100,
        budget_exceeded_days=0,
        card_day_precision=0.50,
        card_day_recall=0.60,
        card_day_f1=0.545,
        true_negative_card_days=780,
        false_positive_card_days=100,
        false_negative_card_days=40,
        true_positive_card_days=60,
    )

    selected = select_model_threshold_policy(
        [
            ModelThresholdCandidate(
                model_name="logistic_regression",
                average_precision=0.40,
                threshold_metrics=logistic_metrics,
            ),
            ModelThresholdCandidate(
                model_name="xgboost",
                average_precision=0.30,
                threshold_metrics=xgboost_metrics,
            ),
        ]
    )

    assert selected.model_name == "xgboost"


def _policy_metrics(
    *,
    card_day_recall: float,
    card_day_precision: float,
) -> CapacityConstrainedThreshold:
    return CapacityConstrainedThreshold(
        threshold=0.85,
        daily_card_capacity=100,
        card_alert_count=200,
        mean_daily_card_alerts=50.0,
        max_daily_card_alerts=100,
        budget_exceeded_days=0,
        card_day_precision=card_day_precision,
        card_day_recall=card_day_recall,
        card_day_f1=0.50,
        true_negative_card_days=800,
        false_positive_card_days=100,
        false_negative_card_days=50,
        true_positive_card_days=50,
    )


def test_select_model_policy_uses_card_precision_as_first_tiebreaker() -> None:
    selected = select_model_threshold_policy(
        [
            ModelThresholdCandidate(
                model_name="logistic_regression",
                average_precision=0.50,
                threshold_metrics=_policy_metrics(
                    card_day_recall=0.60,
                    card_day_precision=0.40,
                ),
            ),
            ModelThresholdCandidate(
                model_name="xgboost",
                average_precision=0.30,
                threshold_metrics=_policy_metrics(
                    card_day_recall=0.60,
                    card_day_precision=0.50,
                ),
            ),
        ]
    )

    assert selected.model_name == "xgboost"


def test_select_model_policy_uses_average_precision_as_second_tiebreaker() -> None:
    selected = select_model_threshold_policy(
        [
            ModelThresholdCandidate(
                model_name="logistic_regression",
                average_precision=0.40,
                threshold_metrics=_policy_metrics(
                    card_day_recall=0.60,
                    card_day_precision=0.50,
                ),
            ),
            ModelThresholdCandidate(
                model_name="xgboost",
                average_precision=0.45,
                threshold_metrics=_policy_metrics(
                    card_day_recall=0.60,
                    card_day_precision=0.50,
                ),
            ),
        ]
    )

    assert selected.model_name == "xgboost"


def test_select_model_policy_rejects_empty_candidates() -> None:
    with pytest.raises(
        ValueError,
        match="candidates must contain at least one model",
    ):
        select_model_threshold_policy([])
