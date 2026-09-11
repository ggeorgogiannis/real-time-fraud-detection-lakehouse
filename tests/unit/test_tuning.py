import pandas as pd
import pytest

from fraud_lakehouse.tuning import (
    build_prequential_folds,
    card_precision_at_k,
    sample_hyperparameter_candidates,
)


def _temporal_partition() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tx_datetime": pd.date_range(
                "2026-01-01",
                periods=14,
                freq="D",
                tz="UTC",
            ),
        }
    )


def test_build_prequential_folds_uses_expanding_training_windows() -> None:
    partition = _temporal_partition()

    folds = build_prequential_folds(
        partition,
        n_folds=2,
        assessment_days=3,
        gap_days=2,
    )

    assert len(folds) == 2

    first, second = folds

    assert first.train_indices == tuple(range(6))
    assert first.assessment_indices == (8, 9, 10)
    assert first.train_end == pd.Timestamp("2026-01-07T00:00:00Z")
    assert first.assessment_start == pd.Timestamp("2026-01-09T00:00:00Z")
    assert first.assessment_end == pd.Timestamp("2026-01-12T00:00:00Z")

    assert second.train_indices == tuple(range(9))
    assert second.assessment_indices == (11, 12, 13)
    assert second.train_end == pd.Timestamp("2026-01-10T00:00:00Z")
    assert second.assessment_start == pd.Timestamp("2026-01-12T00:00:00Z")
    assert second.assessment_end == pd.Timestamp("2026-01-15T00:00:00Z")


@pytest.mark.parametrize(
    ("n_folds", "assessment_days", "gap_days", "message"),
    [
        (0, 3, 2, "n_folds"),
        (2, 0, 2, "assessment_days"),
        (2, 3, -1, "gap_days"),
    ],
)
def test_build_prequential_folds_rejects_invalid_configuration(
    n_folds: int,
    assessment_days: int,
    gap_days: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_prequential_folds(
            _temporal_partition(),
            n_folds=n_folds,
            assessment_days=assessment_days,
            gap_days=gap_days,
        )


def test_build_prequential_folds_requires_timestamp_column() -> None:
    with pytest.raises(ValueError, match="tx_datetime"):
        build_prequential_folds(
            pd.DataFrame({"value": [1, 2, 3]}),
            n_folds=1,
            assessment_days=1,
            gap_days=0,
        )


def test_build_prequential_folds_rejects_insufficient_history() -> None:
    with pytest.raises(ValueError, match="training rows"):
        build_prequential_folds(
            _temporal_partition(),
            n_folds=5,
            assessment_days=3,
            gap_days=2,
        )


def test_card_precision_at_k_groups_customers_and_removes_detected_cards() -> None:
    transactions = pd.DataFrame(
        {
            "transaction_date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-01",
                    "2026-01-01",
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-02",
                    "2026-01-02",
                    "2026-01-02",
                ]
            ).date,
            "customer_id": [1, 1, 2, 3, 1, 2, 3, 4],
            "tx_fraud": [0, 1, 0, 1, 1, 0, 0, 1],
        }
    )
    fraud_probability = [
        0.95,
        0.90,
        0.80,
        0.10,
        0.99,
        0.80,
        0.70,
        0.60,
    ]

    result = card_precision_at_k(
        transactions,
        fraud_probability,
        k=2,
    )

    assert result == pytest.approx(0.25)


def test_card_precision_at_k_rejects_invalid_inputs() -> None:
    partition = pd.DataFrame(
        {
            "transaction_date": [pd.Timestamp("2026-01-01").date()],
            "customer_id": [1],
            "tx_fraud": [1],
        }
    )

    with pytest.raises(ValueError, match="k must be greater than zero"):
        card_precision_at_k(partition, [0.9], k=0)

    with pytest.raises(ValueError, match="Missing required columns: customer_id"):
        card_precision_at_k(
            partition.drop(columns="customer_id"),
            [0.9],
            k=1,
        )

    with pytest.raises(
        ValueError,
        match="partition and fraud_probability must have equal lengths",
    ):
        card_precision_at_k(partition, [], k=1)

    with pytest.raises(
        ValueError,
        match="fraud_probability must contain finite values",
    ):
        card_precision_at_k(partition, [float("nan")], k=1)

    with pytest.raises(
        ValueError,
        match="fraud_probability must contain values between 0 and 1",
    ):
        card_precision_at_k(partition, [1.1], k=1)


def test_sample_hyperparameter_candidates_is_deterministic() -> None:
    first_sample = sample_hyperparameter_candidates(
        "xgboost",
        n_iter=4,
        random_state=17,
    )
    second_sample = sample_hyperparameter_candidates(
        "xgboost",
        n_iter=4,
        random_state=17,
    )

    assert first_sample == second_sample
    assert len(first_sample) == 4
    assert len({tuple(sorted(candidate.items())) for candidate in first_sample}) == 4

    expected_parameters = {
        "colsample_bytree",
        "learning_rate",
        "max_depth",
        "min_child_weight",
        "n_estimators",
        "reg_alpha",
        "reg_lambda",
        "subsample",
    }
    assert all(set(candidate) == expected_parameters for candidate in first_sample)


def test_sample_hyperparameter_candidates_supports_logistic_regression() -> None:
    candidates = sample_hyperparameter_candidates(
        "logistic_regression",
        n_iter=3,
        random_state=42,
    )

    assert len(candidates) == 3
    assert all(set(candidate) == {"C", "class_weight"} for candidate in candidates)


def test_sample_hyperparameter_candidates_rejects_invalid_requests() -> None:
    with pytest.raises(ValueError, match="Unsupported model_name"):
        sample_hyperparameter_candidates(
            "random_forest",
            n_iter=1,
        )

    with pytest.raises(ValueError, match="n_iter must be greater than zero"):
        sample_hyperparameter_candidates(
            "xgboost",
            n_iter=0,
        )
