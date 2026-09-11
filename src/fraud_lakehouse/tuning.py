from dataclasses import dataclass

import pandas as pd


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
