from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler

from fraud_lakehouse.ml_dataset import MODEL_TARGET_COLUMN

_LOGISTIC_REGRESSION_PARAMETER_SPACE = {
    "C": (0.001, 0.01, 0.1, 1.0, 10.0, 100.0),
    "class_weight": (None, "balanced"),
}

_XGBOOST_PARAMETER_SPACE = {
    "n_estimators": (100, 200, 300, 500),
    "max_depth": (3, 4, 6, 8),
    "learning_rate": (0.01, 0.03, 0.05, 0.1, 0.2),
    "min_child_weight": (1, 3, 5, 10),
    "subsample": (0.6, 0.8, 1.0),
    "colsample_bytree": (0.6, 0.8, 1.0),
    "reg_alpha": (0.0, 0.01, 0.1, 1.0),
    "reg_lambda": (0.1, 1.0, 5.0, 10.0),
}

_HYPERPARAMETER_SPACES = {
    "logistic_regression": _LOGISTIC_REGRESSION_PARAMETER_SPACE,
    "xgboost": _XGBOOST_PARAMETER_SPACE,
}


@dataclass(frozen=True)
class PrequentialFold:
    """Positional indices and UTC boundaries for one temporal fold."""

    train_indices: tuple[int, ...]
    assessment_indices: tuple[int, ...]
    train_end: pd.Timestamp
    assessment_start: pd.Timestamp
    assessment_end: pd.Timestamp


def build_prequential_folds(
    partition: pd.DataFrame,
    *,
    n_folds: int,
    assessment_days: int,
    gap_days: int,
) -> tuple[PrequentialFold, ...]:
    """Build expanding temporal folds ending at the partition's final day."""
    if n_folds <= 0:
        raise ValueError("n_folds must be greater than zero.")

    if assessment_days <= 0:
        raise ValueError("assessment_days must be greater than zero.")

    if gap_days < 0:
        raise ValueError("gap_days must be non-negative.")

    if "tx_datetime" not in partition.columns:
        raise ValueError("partition must contain tx_datetime.")

    timestamps = pd.to_datetime(
        partition["tx_datetime"],
        errors="coerce",
        utc=True,
    )

    if timestamps.isna().any():
        raise ValueError("tx_datetime must contain valid timestamps.")

    assessment_delta = pd.Timedelta(days=assessment_days)
    gap_delta = pd.Timedelta(days=gap_days)
    dataset_end = timestamps.max().normalize() + pd.Timedelta(days=1)
    first_assessment_start = dataset_end - (n_folds * assessment_delta)

    folds: list[PrequentialFold] = []

    for fold_index in range(n_folds):
        assessment_start = first_assessment_start + (fold_index * assessment_delta)
        assessment_end = assessment_start + assessment_delta
        train_end = assessment_start - gap_delta

        train_mask = timestamps < train_end
        assessment_mask = (timestamps >= assessment_start) & (timestamps < assessment_end)

        train_indices = tuple(int(index) for index in train_mask.to_numpy().nonzero()[0])
        assessment_indices = tuple(int(index) for index in assessment_mask.to_numpy().nonzero()[0])

        if not train_indices:
            raise ValueError("Each prequential fold must contain training rows.")

        if not assessment_indices:
            raise ValueError("Each prequential fold must contain assessment rows.")

        folds.append(
            PrequentialFold(
                train_indices=train_indices,
                assessment_indices=assessment_indices,
                train_end=train_end,
                assessment_start=assessment_start,
                assessment_end=assessment_end,
            )
        )

    return tuple(folds)


def sample_hyperparameter_candidates(
    model_name: str,
    *,
    n_iter: int,
    random_state: int = 42,
) -> tuple[dict[str, object], ...]:
    """Sample deterministic hyperparameter candidates for a supported model."""
    if model_name not in _HYPERPARAMETER_SPACES:
        supported = ", ".join(sorted(_HYPERPARAMETER_SPACES))
        raise ValueError(f"Unsupported model_name: {model_name}. Supported models: {supported}.")

    if n_iter <= 0:
        raise ValueError("n_iter must be greater than zero.")

    sampled = ParameterSampler(
        _HYPERPARAMETER_SPACES[model_name],
        n_iter=n_iter,
        random_state=random_state,
    )

    return tuple(dict(candidate) for candidate in sampled)


def card_precision_at_k(
    partition: pd.DataFrame,
    fraud_probability: Sequence[float],
    *,
    k: int,
) -> float:
    """Calculate mean daily Card Precision at k with detected-card removal."""
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    required_columns = {
        "transaction_date",
        "customer_id",
        MODEL_TARGET_COLUMN,
    }
    missing_columns = sorted(required_columns.difference(partition.columns))

    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns: {missing}")

    if partition.empty:
        raise ValueError("partition must contain at least one row.")

    transaction_dates = pd.to_datetime(
        partition["transaction_date"],
        errors="coerce",
    )
    if transaction_dates.isna().any():
        raise ValueError("transaction_date must contain valid dates.")

    if partition["customer_id"].isna().any():
        raise ValueError("customer_id must not contain missing values.")

    target = pd.to_numeric(
        partition[MODEL_TARGET_COLUMN],
        errors="coerce",
    )
    if target.isna().any() or not target.isin((0, 1)).all():
        raise ValueError("tx_fraud must contain only 0 or 1.")

    probability_values = np.asarray(
        fraud_probability,
        dtype=float,
    )
    if probability_values.ndim != 1:
        raise ValueError("fraud_probability must be one-dimensional.")

    if len(probability_values) != len(partition):
        raise ValueError("partition and fraud_probability must have equal lengths.")

    if not np.isfinite(probability_values).all():
        raise ValueError("fraud_probability must contain finite values.")

    if ((probability_values < 0.0) | (probability_values > 1.0)).any():
        raise ValueError("fraud_probability must contain values between 0 and 1.")

    predictions = pd.DataFrame(
        {
            "transaction_date": transaction_dates.dt.date.to_numpy(),
            "customer_id": partition["customer_id"].to_numpy(),
            MODEL_TARGET_COLUMN: target.astype("int8").to_numpy(),
            "fraud_probability": probability_values,
        }
    )

    detected_cards: set[object] = set()
    daily_precision: list[float] = []

    for transaction_date in sorted(predictions["transaction_date"].unique()):
        daily_transactions = predictions.loc[
            predictions["transaction_date"].eq(transaction_date)
            & ~predictions["customer_id"].isin(detected_cards)
        ]

        daily_cards = (
            daily_transactions.groupby(
                "customer_id",
                as_index=False,
                sort=False,
            )
            .agg(
                {
                    MODEL_TARGET_COLUMN: "max",
                    "fraud_probability": "max",
                }
            )
            .sort_values(
                "fraud_probability",
                ascending=False,
                kind="mergesort",
            )
            .head(k)
        )

        compromised_cards = daily_cards.loc[
            daily_cards[MODEL_TARGET_COLUMN].eq(1),
            "customer_id",
        ].tolist()

        daily_precision.append(len(compromised_cards) / k)
        detected_cards.update(compromised_cards)

    return float(np.mean(daily_precision))
