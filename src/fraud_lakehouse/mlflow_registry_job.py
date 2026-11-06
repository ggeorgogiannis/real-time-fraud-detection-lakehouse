import argparse
import logging
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from fraud_lakehouse.ml_dataset import MODEL_FEATURE_COLUMNS
from fraud_lakehouse.mlflow_registry import (
    DEFAULT_MLFLOW_EXPERIMENT,
    DEFAULT_MODEL_ALIAS,
    DEFAULT_REGISTERED_MODEL_NAME,
    publish_approved_fraud_policy,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_MODEL_DIR = Path("data/models")
DEFAULT_POLICY_PATH = DEFAULT_MODEL_DIR / "threshold_policy.json"
DEFAULT_EVALUATION_PATH = DEFAULT_MODEL_DIR / "final_evaluation.json"
DEFAULT_INPUT_EXAMPLE_PATH = Path("data/ml/test.parquet")
DEFAULT_INPUT_EXAMPLE_ROWS = 5

LOG_LEVELS = (
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
)


def build_parser() -> argparse.ArgumentParser:
    """Create the approved-policy registration parser."""
    parser = argparse.ArgumentParser(
        description=("Validate and register the approved fraud policy with MLflow.")
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv(
            "MLFLOW_TRACKING_URI",
            DEFAULT_TRACKING_URI,
        ),
        help="MLflow tracking server URI.",
    )
    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_MLFLOW_EXPERIMENT,
        help="MLflow experiment used for policy registration.",
    )
    parser.add_argument(
        "--registered-model-name",
        default=DEFAULT_REGISTERED_MODEL_NAME,
        help="MLflow registered-model name.",
    )
    parser.add_argument(
        "--alias",
        default=DEFAULT_MODEL_ALIAS,
        help="Alias assigned after prediction verification.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory containing the selected tuned model.",
    )
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=DEFAULT_POLICY_PATH,
        help="Locked validation threshold policy.",
    )
    parser.add_argument(
        "--evaluation-path",
        type=Path,
        default=DEFAULT_EVALUATION_PATH,
        help="Final test evaluation for the locked policy.",
    )
    parser.add_argument(
        "--input-example-path",
        type=Path,
        default=DEFAULT_INPUT_EXAMPLE_PATH,
        help="Parquet partition used to build the model signature.",
    )
    parser.add_argument(
        "--input-example-rows",
        type=int,
        default=DEFAULT_INPUT_EXAMPLE_ROWS,
        help="Number of model-input rows logged with the model.",
    )
    parser.add_argument(
        "--source-revision",
        default=os.getenv("MODEL_SOURCE_REVISION"),
        help=("Source revision recorded with the model. Defaults to the current Git commit."),
    )
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default="INFO",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
) -> None:
    """Register the approved policy and promote its verified alias."""
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format=("%(asctime)s %(levelname)s %(name)s %(message)s"),
    )

    if args.input_example_rows <= 0:
        raise ValueError("input_example_rows must be greater than zero")

    input_example_path = Path(args.input_example_path)
    if not input_example_path.is_file():
        raise FileNotFoundError(f"Model input example partition not found: {input_example_path}")

    input_example = pd.read_parquet(
        input_example_path,
        columns=list(MODEL_FEATURE_COLUMNS),
        engine="pyarrow",
    ).head(args.input_example_rows)

    if input_example.empty:
        raise ValueError("Model input example partition must not be empty")

    source_revision = _resolve_source_revision(args.source_revision)

    registered = publish_approved_fraud_policy(
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        registered_model_name=args.registered_model_name,
        alias=args.alias,
        model_dir=Path(args.model_dir),
        policy_path=Path(args.policy_path),
        evaluation_path=Path(args.evaluation_path),
        input_example=input_example,
        source_revision=source_revision,
    )

    LOGGER.info(
        "fraud_policy_registered run_id=%s model=%s version=%s alias=%s uri=%s",
        registered.run_id,
        registered.registered_model_name,
        registered.registered_model_version,
        registered.alias,
        registered.registered_model_uri,
    )


def _resolve_source_revision(
    supplied_revision: str | None,
) -> str:
    if supplied_revision is not None:
        revision = supplied_revision.strip()
        if revision:
            return revision

    try:
        result = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            "source_revision must be supplied when the current Git revision cannot be resolved"
        ) from exc

    revision = result.stdout.strip()
    if not revision:
        raise ValueError("Git returned an empty source revision")

    return revision


if __name__ == "__main__":
    main()
