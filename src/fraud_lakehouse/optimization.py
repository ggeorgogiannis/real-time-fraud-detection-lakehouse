import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import xgboost
from sklearn.pipeline import Pipeline

from fraud_lakehouse.ml_dataset import (
    MODEL_FEATURE_COLUMNS,
    MODEL_TARGET_COLUMN,
)
from fraud_lakehouse.modeling import (
    BaselineEstimator,
    build_model_preprocessor,
    extract_model_inputs,
)
from fraud_lakehouse.tuning import (
    HyperparameterSearchResult,
    build_prequential_folds,
    build_tuning_estimator,
    run_hyperparameter_search,
)


@dataclass(frozen=True)
class TunedModelArtifact:
    """Preprocessing and estimator fitted with selected hyperparameters."""

    preprocessor: Pipeline
    estimator: BaselineEstimator
    feature_columns: tuple[str, ...]
    target_column: str
    hyperparameters: dict[str, object]


@dataclass(frozen=True)
class HyperparameterOptimizationOutputs:
    """Paths to tuned models and their reproducible search results."""

    logistic_model_path: Path
    xgboost_model_path: Path
    results_path: Path


def optimize_and_publish_hyperparameters(
    *,
    dataset_dir: Path,
    output_dir: Path,
    logistic_iterations: int,
    xgboost_iterations: int,
    n_folds: int,
    assessment_days: int,
    gap_days: int,
    card_precision_k: int,
    random_state: int = 42,
) -> HyperparameterOptimizationOutputs:
    """Tune models on training data and publish refitted winners."""
    training_path = Path(dataset_dir) / "train.parquet"

    if not training_path.is_file():
        raise FileNotFoundError(f"Training dataset partition not found: {training_path}")

    training_partition = pd.read_parquet(
        training_path,
        engine="pyarrow",
    )
    folds = build_prequential_folds(
        training_partition,
        n_folds=n_folds,
        assessment_days=assessment_days,
        gap_days=gap_days,
    )

    searches = {
        "logistic_regression": run_hyperparameter_search(
            training_partition,
            folds,
            model_name="logistic_regression",
            n_iter=logistic_iterations,
            card_precision_k=card_precision_k,
            random_state=random_state,
        ),
        "xgboost": run_hyperparameter_search(
            training_partition,
            folds,
            model_name="xgboost",
            n_iter=xgboost_iterations,
            card_precision_k=card_precision_k,
            random_state=random_state,
        ),
    }

    artifacts = {
        model_name: _fit_selected_model(
            training_partition,
            search,
            random_state=random_state,
        )
        for model_name, search in searches.items()
    }

    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    logistic_model_path = output_directory / "tuned_logistic_regression.joblib"
    xgboost_model_path = output_directory / "tuned_xgboost.joblib"
    results_path = output_directory / "hyperparameter_search.json"

    model_paths = {
        "logistic_regression": logistic_model_path,
        "xgboost": xgboost_model_path,
    }
    temporary_model_paths = {
        model_name: output_directory / f".tuned_{model_name}.joblib.tmp"
        for model_name in model_paths
    }
    temporary_results_path = output_directory / ".hyperparameter_search.json.tmp"

    results = {
        "schema_version": 1,
        "selection_data": "train",
        "primary_metric": "average_precision",
        "secondary_metric": "card_precision_at_k",
        "card_precision_k": card_precision_k,
        "random_state": random_state,
        "threshold_optimization": {
            "performed": False,
        },
        "validation_strategy": {
            "type": "prequential_expanding_window",
            "n_folds": n_folds,
            "assessment_days": assessment_days,
            "gap_days": gap_days,
        },
        "libraries": {
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "models": {
            model_name: _search_result_payload(search) for model_name, search in searches.items()
        },
    }

    try:
        for model_name, artifact in artifacts.items():
            joblib.dump(
                artifact,
                temporary_model_paths[model_name],
            )

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

        for model_name, model_path in model_paths.items():
            temporary_model_paths[model_name].replace(model_path)

        # Publish results last so they describe a complete artifact set.
        temporary_results_path.replace(results_path)
    finally:
        for temporary_path in (
            *temporary_model_paths.values(),
            temporary_results_path,
        ):
            temporary_path.unlink(missing_ok=True)

    return HyperparameterOptimizationOutputs(
        logistic_model_path=logistic_model_path,
        xgboost_model_path=xgboost_model_path,
        results_path=results_path,
    )


def _fit_selected_model(
    training_partition: pd.DataFrame,
    search: HyperparameterSearchResult,
    *,
    random_state: int,
) -> TunedModelArtifact:
    training_inputs = extract_model_inputs(training_partition)

    if training_inputs.target.nunique() != 2:
        raise ValueError("The full training target must contain both fraud classes.")

    preprocessor = build_model_preprocessor()
    transformed_features = preprocessor.fit_transform(
        training_inputs.features,
        training_inputs.target,
    )

    hyperparameters = dict(search.best_candidate.parameters)
    estimator = build_tuning_estimator(
        model_name=search.model_name,
        parameters=hyperparameters,
        training_target=training_inputs.target,
        random_state=random_state,
    )
    estimator.fit(
        transformed_features,
        training_inputs.target,
    )

    return TunedModelArtifact(
        preprocessor=preprocessor,
        estimator=estimator,
        feature_columns=MODEL_FEATURE_COLUMNS,
        target_column=MODEL_TARGET_COLUMN,
        hyperparameters=hyperparameters,
    )


def _search_result_payload(
    search: HyperparameterSearchResult,
) -> dict[str, object]:
    return {
        "candidate_count": len(search.candidates),
        "best_candidate": asdict(search.best_candidate),
        "candidates": [asdict(candidate) for candidate in search.candidates],
    }
