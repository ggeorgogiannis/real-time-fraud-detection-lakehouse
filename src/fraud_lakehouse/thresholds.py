from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraud_lakehouse.ml_dataset import MODEL_TARGET_COLUMN


@dataclass(frozen=True)
class CapacityConstrainedThreshold:
    """Card-day metrics for a threshold and daily alert capacity."""

    threshold: float
    daily_card_capacity: int
    card_alert_count: int
    mean_daily_card_alerts: float
    max_daily_card_alerts: int
    budget_exceeded_days: int
    card_day_precision: float
    card_day_recall: float
    card_day_f1: float
    true_negative_card_days: int
    false_positive_card_days: int
    false_negative_card_days: int
    true_positive_card_days: int


@dataclass(frozen=True)
class ModelThresholdCandidate:
    """Validation result for one model and its selected threshold."""

    model_name: str
    average_precision: float
    threshold_metrics: CapacityConstrainedThreshold


def select_model_threshold_policy(
    candidates: Sequence[ModelThresholdCandidate],
) -> ModelThresholdCandidate:
    """Select the validation policy with the strongest card-day recall."""
    candidate_list = tuple(candidates)

    if not candidate_list:
        raise ValueError("candidates must contain at least one model.")

    model_names = [candidate.model_name for candidate in candidate_list]
    if any(not model_name.strip() for model_name in model_names):
        raise ValueError("model_name must not be empty.")

    if len(model_names) != len(set(model_names)):
        raise ValueError("model_name values must be unique.")

    for candidate in candidate_list:
        if not np.isfinite(candidate.average_precision) or not (
            0.0 <= candidate.average_precision <= 1.0
        ):
            raise ValueError("average_precision must be between 0 and 1.")

        if candidate.threshold_metrics.budget_exceeded_days:
            raise ValueError("All threshold candidates must respect daily capacity.")

    return min(
        candidate_list,
        key=lambda candidate: (
            -candidate.threshold_metrics.card_day_recall,
            -candidate.threshold_metrics.card_day_precision,
            -candidate.average_precision,
            candidate.model_name,
        ),
    )


def select_capacity_constrained_threshold(
    partition: pd.DataFrame,
    fraud_probability: Sequence[float],
    *,
    daily_card_capacity: int,
) -> CapacityConstrainedThreshold:
    """Choose the lowest threshold that respects capacity every day."""
    _validate_daily_card_capacity(daily_card_capacity)
    card_days = _build_card_days(
        partition,
        fraud_probability,
    )

    capacity_boundaries = []

    for _, daily_cards in card_days.groupby(
        "transaction_date",
        sort=True,
    ):
        if len(daily_cards) <= daily_card_capacity:
            continue

        descending_probability = np.sort(daily_cards["fraud_probability"].to_numpy())[::-1]
        capacity_boundaries.append(descending_probability[daily_card_capacity])

    if capacity_boundaries:
        threshold = float(
            np.nextafter(
                max(capacity_boundaries),
                np.inf,
            )
        )
    else:
        threshold = 0.0

    if threshold > 1.0:
        raise ValueError(
            "No threshold between 0 and 1 can satisfy the daily "
            "card capacity because too many cards share "
            "probability 1."
        )

    return _evaluate_card_days(
        card_days,
        threshold=threshold,
        daily_card_capacity=daily_card_capacity,
    )


def evaluate_card_day_threshold(
    partition: pd.DataFrame,
    fraud_probability: Sequence[float],
    *,
    threshold: float,
    daily_card_capacity: int,
) -> CapacityConstrainedThreshold:
    """Evaluate a fixed threshold without selecting a new one."""
    _validate_daily_card_capacity(daily_card_capacity)

    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    card_days = _build_card_days(
        partition,
        fraud_probability,
    )

    return _evaluate_card_days(
        card_days,
        threshold=float(threshold),
        daily_card_capacity=daily_card_capacity,
    )


def _validate_daily_card_capacity(
    daily_card_capacity: int,
) -> None:
    if daily_card_capacity <= 0:
        raise ValueError("daily_card_capacity must be greater than zero.")


def _build_card_days(
    partition: pd.DataFrame,
    fraud_probability: Sequence[float],
) -> pd.DataFrame:
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
            "transaction_date": (transaction_dates.dt.date.to_numpy()),
            "customer_id": partition["customer_id"].to_numpy(),
            MODEL_TARGET_COLUMN: (target.astype("int8").to_numpy()),
            "fraud_probability": probability_values,
        }
    )

    return (
        predictions.groupby(
            ["transaction_date", "customer_id"],
            as_index=False,
            sort=True,
        )
        .agg(
            {
                MODEL_TARGET_COLUMN: "max",
                "fraud_probability": "max",
            }
        )
        .sort_values(
            ["transaction_date", "customer_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def _evaluate_card_days(
    card_days: pd.DataFrame,
    *,
    threshold: float,
    daily_card_capacity: int,
) -> CapacityConstrainedThreshold:
    evaluated_card_days = card_days.copy()
    evaluated_card_days["is_alert"] = evaluated_card_days["fraud_probability"].ge(threshold)

    daily_alerts = evaluated_card_days.groupby(
        "transaction_date",
        sort=True,
    )["is_alert"].sum()

    actual = evaluated_card_days[MODEL_TARGET_COLUMN].eq(1)
    predicted = evaluated_card_days["is_alert"]

    true_positives = int((actual & predicted).sum())
    false_positives = int((~actual & predicted).sum())
    false_negatives = int((actual & ~predicted).sum())
    true_negatives = int((~actual & ~predicted).sum())

    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives

    precision = true_positives / precision_denominator if precision_denominator else 0.0
    recall = true_positives / recall_denominator if recall_denominator else 0.0
    f1_score = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    return CapacityConstrainedThreshold(
        threshold=threshold,
        daily_card_capacity=daily_card_capacity,
        card_alert_count=int(daily_alerts.sum()),
        mean_daily_card_alerts=float(daily_alerts.mean()),
        max_daily_card_alerts=int(daily_alerts.max()),
        budget_exceeded_days=int(daily_alerts.gt(daily_card_capacity).sum()),
        card_day_precision=float(precision),
        card_day_recall=float(recall),
        card_day_f1=float(f1_score),
        true_negative_card_days=true_negatives,
        false_positive_card_days=false_positives,
        false_negative_card_days=false_negatives,
        true_positive_card_days=true_positives,
    )
