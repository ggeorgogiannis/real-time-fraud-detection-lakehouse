import pandas as pd
import pytest

from fraud_lakehouse.tuning import build_prequential_folds


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
