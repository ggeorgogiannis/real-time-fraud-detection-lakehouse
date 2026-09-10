import numpy as np
import pandas as pd
import pytest

from fraud_lakehouse.ml_dataset import (
    MODEL_FEATURE_COLUMNS,
    MODEL_TARGET_COLUMN,
)
from fraud_lakehouse.modeling import (
    evaluate_binary_classifier,
    prepare_model_partitions,
    train_baseline_models,
)


def _partition(
    amounts: list[float],
    fraud_labels: list[int],
) -> pd.DataFrame:
    row_count = len(amounts)
    previous_means = [None if index == 0 else amounts[index - 1] for index in range(row_count)]
    amount_ratios = [
        None if previous_mean is None else amount / previous_mean
        for amount, previous_mean in zip(
            amounts,
            previous_means,
            strict=True,
        )
    ]

    return pd.DataFrame(
        {
            "transaction_id": range(1, row_count + 1),
            "tx_amount": amounts,
            "transaction_hour": range(row_count),
            "day_of_week": range(row_count),
            "is_weekend": [0] * row_count,
            "is_night": [1, *([0] * (row_count - 1))],
            "customer_previous_transaction_count": range(row_count),
            "customer_previous_mean_amount": previous_means,
            "amount_to_customer_previous_mean": amount_ratios,
            "terminal_previous_transaction_count": range(row_count),
            MODEL_TARGET_COLUMN: fraud_labels,
        }
    )


def test_prepare_model_partitions_selects_features_and_target() -> None:
    train = _partition([10.0, 20.0, 30.0], [0, 0, 1])
    validation = _partition([40.0, 50.0], [0, 1])
    test = _partition([60.0, 70.0], [1, 0])

    prepared = prepare_model_partitions(
        train=train,
        validation=validation,
        test=test,
    )

    assert prepared.train.features.columns.tolist() == list(MODEL_FEATURE_COLUMNS)
    assert prepared.validation.features.columns.tolist() == list(MODEL_FEATURE_COLUMNS)
    assert prepared.test.features.columns.tolist() == list(MODEL_FEATURE_COLUMNS)

    assert prepared.train.target.tolist() == [0, 0, 1]
    assert prepared.validation.target.tolist() == [0, 1]
    assert prepared.test.target.tolist() == [1, 0]


def test_prepare_model_partitions_imputes_missing_historical_values() -> None:
    prepared = prepare_model_partitions(
        train=_partition([10.0, 20.0, 30.0], [0, 0, 1]),
        validation=_partition([40.0, 50.0], [0, 1]),
        test=_partition([60.0, 70.0], [1, 0]),
    )

    for partition in (
        prepared.train,
        prepared.validation,
        prepared.test,
    ):
        values = partition.features.to_numpy()

        assert not partition.features.isna().any().any()
        assert np.isfinite(values).all()


def test_prepare_model_partitions_fits_preprocessor_on_training_only() -> None:
    prepared = prepare_model_partitions(
        train=_partition([10.0, 20.0, 30.0], [0, 0, 1]),
        validation=_partition([10_000.0, 20_000.0], [0, 1]),
        test=_partition([30_000.0, 40_000.0], [1, 0]),
    )

    amount_index = list(MODEL_FEATURE_COLUMNS).index("tx_amount")
    scaler = prepared.preprocessor.named_steps["scaler"]

    assert scaler.mean_[amount_index] == pytest.approx(20.0)


def test_prepare_model_partitions_rejects_missing_columns() -> None:
    train = _partition([10.0, 20.0, 30.0], [0, 0, 1]).drop(columns="tx_amount")

    with pytest.raises(
        ValueError,
        match="Missing required columns: tx_amount",
    ):
        prepare_model_partitions(
            train=train,
            validation=_partition([40.0, 50.0], [0, 1]),
            test=_partition([60.0, 70.0], [1, 0]),
        )


def test_evaluate_binary_classifier_calculates_imbalanced_metrics() -> None:
    metrics = evaluate_binary_classifier(
        target=pd.Series([0, 0, 1, 1]),
        fraud_probability=np.array([0.1, 0.7, 0.8, 0.4]),
        threshold=0.5,
    )

    assert metrics.average_precision == pytest.approx(5.0 / 6.0)
    assert metrics.roc_auc == pytest.approx(0.75)
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1_score == pytest.approx(0.5)
    assert metrics.true_negatives == 1
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.true_positives == 1


def test_train_baseline_models_evaluates_dummy_and_logistic_models() -> None:
    prepared = prepare_model_partitions(
        train=_partition(
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            [0, 0, 0, 1, 0, 1],
        ),
        validation=_partition(
            [15.0, 25.0, 45.0, 55.0],
            [0, 0, 1, 1],
        ),
        test=_partition(
            [12.0, 22.0, 42.0, 62.0],
            [0, 0, 1, 1],
        ),
    )

    evaluations = train_baseline_models(prepared)

    assert [evaluation.name for evaluation in evaluations] == [
        "dummy_prior",
        "logistic_regression",
    ]

    logistic_evaluation = evaluations[1]
    assert logistic_evaluation.estimator.class_weight == "balanced"

    for evaluation in evaluations:
        for metrics in (
            evaluation.validation_metrics,
            evaluation.test_metrics,
        ):
            assert 0.0 <= metrics.average_precision <= 1.0
            assert 0.0 <= metrics.roc_auc <= 1.0
            assert 0.0 <= metrics.precision <= 1.0
            assert 0.0 <= metrics.recall <= 1.0
            assert 0.0 <= metrics.f1_score <= 1.0

            classified_rows = (
                metrics.true_negatives
                + metrics.false_positives
                + metrics.false_negatives
                + metrics.true_positives
            )
            assert classified_rows == 4


def test_evaluate_binary_classifier_rejects_invalid_probabilities() -> None:
    with pytest.raises(
        ValueError,
        match="fraud_probability must contain values between 0 and 1",
    ):
        evaluate_binary_classifier(
            target=pd.Series([0, 1]),
            fraud_probability=np.array([0.2, 1.2]),
            threshold=0.5,
        )
