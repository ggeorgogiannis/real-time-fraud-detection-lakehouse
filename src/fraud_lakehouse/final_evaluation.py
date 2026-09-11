import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from fraud_lakehouse.modeling import (
    evaluate_binary_classifier,
    extract_model_inputs,
    fraud_probability,
)
from fraud_lakehouse.threshold_optimization import (
    load_tuned_artifact,
)
from fraud_lakehouse.thresholds import evaluate_card_day_threshold


@dataclass(frozen=True)
class FinalEvaluationOutputs:
    """Final test metrics for the locked operating policy."""

    results_path: Path
    model_name: str
    threshold: float


def evaluate_and_publish_final_policy(
    *,
    dataset_dir: Path,
    model_dir: Path,
    policy_path: Path,
    output_dir: Path,
) -> FinalEvaluationOutputs:
    """Evaluate the locked model and threshold on the test period."""
    test_path = Path(dataset_dir) / "test.parquet"
    locked_policy_path = Path(policy_path)

    if not test_path.is_file():
        raise FileNotFoundError(f"Test dataset partition not found: {test_path}")

    if not locked_policy_path.is_file():
        raise FileNotFoundError(f"Threshold policy not found: {locked_policy_path}")

    (
        model_name,
        model_artifact_name,
        threshold,
        daily_card_capacity,
    ) = _read_locked_policy(locked_policy_path)

    model_path = Path(model_dir) / model_artifact_name
    if not model_path.is_file():
        raise FileNotFoundError(f"Selected model artifact not found: {model_path}")

    test_partition = pd.read_parquet(
        test_path,
        engine="pyarrow",
    )
    test_inputs = extract_model_inputs(test_partition)
    artifact = load_tuned_artifact(model_path)

    transformed_features = artifact.preprocessor.transform(test_inputs.features)
    probabilities = fraud_probability(
        artifact.estimator,
        transformed_features,
    )

    transaction_metrics = evaluate_binary_classifier(
        target=test_inputs.target,
        fraud_probability=probabilities,
        threshold=threshold,
    )
    card_day_metrics = evaluate_card_day_threshold(
        test_partition,
        probabilities,
        threshold=threshold,
        daily_card_capacity=daily_card_capacity,
    )

    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    results_path = output_directory / "final_evaluation.json"
    temporary_results_path = output_directory / ".final_evaluation.json.tmp"

    results = {
        "schema_version": 1,
        "evaluation_data": "test",
        "selection_locked": True,
        "policy_source": locked_policy_path.name,
        "model_name": model_name,
        "model_artifact": model_artifact_name,
        "threshold": threshold,
        "operating_constraint": {
            "daily_card_capacity": daily_card_capacity,
            "scope": "unique_customer_cards_per_day",
        },
        "transaction_metrics": asdict(transaction_metrics),
        "card_day_metrics": asdict(card_day_metrics),
    }

    try:
        temporary_results_path.write_text(
            json.dumps(
                results,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary_results_path.replace(results_path)
    finally:
        temporary_results_path.unlink(missing_ok=True)

    return FinalEvaluationOutputs(
        results_path=results_path,
        model_name=model_name,
        threshold=threshold,
    )


def _read_locked_policy(
    policy_path: Path,
) -> tuple[str, str, float, int]:
    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Threshold policy is not valid JSON: {policy_path}") from exc

    if not isinstance(policy, dict):
        raise ValueError("Threshold policy must contain a JSON object.")

    if policy.get("schema_version") != 1:
        raise ValueError("Threshold policy must use schema version 1.")

    if policy.get("selection_data") != "validation":
        raise ValueError("Threshold policy must be selected from validation data.")

    if policy.get("test_evaluation") != {"performed": False}:
        raise ValueError("Threshold policy must precede final test evaluation.")

    selected_policy = policy.get("selected_policy")
    if not isinstance(selected_policy, dict):
        raise ValueError("Threshold policy must contain selected_policy.")

    model_name = selected_policy.get("model_name")
    model_artifact_name = selected_policy.get("model_artifact")

    if not isinstance(model_name, str) or not model_name:
        raise ValueError("Selected policy must contain a model name.")

    if (
        not isinstance(model_artifact_name, str)
        or not model_artifact_name
        or Path(model_artifact_name).name != model_artifact_name
    ):
        raise ValueError("Selected policy must contain a model artifact filename.")

    try:
        threshold = float(selected_policy["threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Selected policy must contain a numeric threshold.") from exc

    if not 0.0 < threshold < 1.0:
        raise ValueError("Selected threshold must be between 0 and 1.")

    operating_constraint = policy.get("operating_constraint")
    if not isinstance(operating_constraint, dict):
        raise ValueError("Threshold policy must contain an operating constraint.")

    try:
        daily_card_capacity = int(operating_constraint["daily_card_capacity"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Operating constraint must contain a daily card capacity.") from exc

    if daily_card_capacity <= 0:
        raise ValueError("Daily card capacity must be greater than zero.")

    return (
        model_name,
        model_artifact_name,
        threshold,
        daily_card_capacity,
    )
