from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
from typing import Any, Literal

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response, status
from mlflow import MlflowClient
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from fraud_lakehouse.ml_dataset import MODEL_FEATURE_COLUMNS
from fraud_lakehouse.mlflow_registry import (
    DEFAULT_MODEL_ALIAS,
    DEFAULT_REGISTERED_MODEL_NAME,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_MAX_BATCH_SIZE = 100

APPROVAL_STATUS = "approved"
CAPACITY_SCOPE = "unique_customer_cards_per_day"

OBSERVED_ENDPOINTS = {
    "/health/live",
    "/health/ready",
    "/v1/score",
}

MODEL_INPUT_DTYPES = {
    "tx_amount": "float64",
    "transaction_hour": "int32",
    "day_of_week": "int32",
    "is_weekend": "int32",
    "is_night": "int32",
    "customer_previous_transaction_count": "int64",
    "customer_previous_mean_amount": "float64",
    "amount_to_customer_previous_mean": "float64",
    "terminal_previous_transaction_count": "int64",
}


@dataclass(frozen=True)
class ScoringSettings:
    """Runtime configuration for the fraud-scoring service."""

    tracking_uri: str
    registered_model_name: str
    model_alias: str
    max_batch_size: int

    @classmethod
    def from_environment(cls) -> ScoringSettings:
        """Build validated settings from environment variables."""
        max_batch_size_raw = os.getenv(
            "FRAUD_SCORING_MAX_BATCH_SIZE",
            str(DEFAULT_MAX_BATCH_SIZE),
        )

        try:
            max_batch_size = int(max_batch_size_raw)
        except ValueError as exc:
            raise ValueError("FRAUD_SCORING_MAX_BATCH_SIZE must be an integer") from exc

        settings = cls(
            tracking_uri=os.getenv(
                "MLFLOW_TRACKING_URI",
                DEFAULT_TRACKING_URI,
            ),
            registered_model_name=os.getenv(
                "FRAUD_REGISTERED_MODEL_NAME",
                DEFAULT_REGISTERED_MODEL_NAME,
            ),
            model_alias=os.getenv(
                "FRAUD_MODEL_ALIAS",
                DEFAULT_MODEL_ALIAS,
            ),
            max_batch_size=max_batch_size,
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        """Validate settings before loading a registered model."""
        for name, value in (
            ("tracking_uri", self.tracking_uri),
            (
                "registered_model_name",
                self.registered_model_name,
            ),
            ("model_alias", self.model_alias),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be empty")

        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be greater than zero")


@dataclass(frozen=True)
class RegisteredModelMetadata:
    """Approved concrete registry version loaded by the service."""

    model_name: str
    model_version: str
    model_alias: str
    model_uri: str
    source_revision: str
    threshold: float
    daily_card_capacity: int
    capacity_scope: str


@dataclass(frozen=True)
class ScoringRuntime:
    """Loaded PyFunc model and its immutable registry metadata."""

    model: Any
    metadata: RegisteredModelMetadata


@dataclass(frozen=True)
class ScoringMetrics:
    """Prometheus collectors owned by one API instance."""

    requests_total: Counter
    request_duration_seconds: Histogram
    scored_transactions_total: Counter
    threshold_decisions_total: Counter
    fraud_probability: Histogram
    model_info: Gauge

    @classmethod
    def create(
        cls,
        registry: CollectorRegistry,
    ) -> ScoringMetrics:
        return cls(
            requests_total=Counter(
                "fraud_scoring_requests_total",
                "HTTP requests handled by the scoring API.",
                labelnames=(
                    "endpoint",
                    "method",
                    "status_code",
                ),
                registry=registry,
            ),
            request_duration_seconds=Histogram(
                "fraud_scoring_request_duration_seconds",
                "Scoring API request duration in seconds.",
                labelnames=(
                    "endpoint",
                    "method",
                ),
                registry=registry,
            ),
            scored_transactions_total=Counter(
                "fraud_scoring_transactions_total",
                "Transactions scored by the approved model.",
                registry=registry,
            ),
            threshold_decisions_total=Counter(
                "fraud_scoring_threshold_decisions_total",
                ("Transactions whose probability met the locked threshold."),
                registry=registry,
            ),
            fraud_probability=Histogram(
                "fraud_scoring_probability",
                "Distribution of predicted fraud probabilities.",
                buckets=(
                    0.0,
                    0.05,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    0.95,
                    1.0,
                ),
                registry=registry,
            ),
            model_info=Gauge(
                "fraud_scoring_model_info",
                "Currently loaded approved model version.",
                labelnames=(
                    "model_name",
                    "model_version",
                    "model_alias",
                    "source_revision",
                ),
                registry=registry,
            ),
        )


class TransactionFeatures(BaseModel):
    """Identity and point-in-time features for one transaction."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )

    transaction_id: int = Field(gt=0)
    tx_datetime: datetime = Field(strict=False)
    customer_id: int = Field(gt=0)
    terminal_id: int = Field(gt=0)

    tx_amount: float = Field(ge=0.0)
    transaction_hour: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    is_weekend: Literal[0, 1]
    is_night: Literal[0, 1]
    customer_previous_transaction_count: int = Field(ge=0)
    customer_previous_mean_amount: float | None = Field(
        default=None,
        ge=0.0,
    )
    amount_to_customer_previous_mean: float | None = Field(
        default=None,
        ge=0.0,
    )
    terminal_previous_transaction_count: int = Field(ge=0)

    @field_validator("tx_datetime")
    @classmethod
    def require_timezone(
        cls,
        value: datetime,
    ) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("tx_datetime must include a timezone")
        return value


class ScoreRequest(BaseModel):
    """Bounded batch of transactions to score."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )

    transactions: list[TransactionFeatures] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_transaction_ids(
        self,
    ) -> ScoreRequest:
        transaction_ids = [transaction.transaction_id for transaction in self.transactions]

        if len(transaction_ids) != len(set(transaction_ids)):
            raise ValueError("transaction_id values must be unique within one request")

        return self


class TransactionPrediction(BaseModel):
    """Prediction corresponding to one submitted transaction."""

    model_config = ConfigDict(extra="forbid")

    transaction_id: int
    customer_id: int
    terminal_id: int
    fraud_probability: float = Field(ge=0.0, le=1.0)
    threshold_decision: bool


class ModelResponseMetadata(BaseModel):
    """Registry and operating-policy metadata returned to clients."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    alias: str
    source_revision: str
    threshold: float = Field(gt=0.0, lt=1.0)
    daily_card_capacity: int = Field(gt=0)
    capacity_scope: str
    capacity_enforced_by_endpoint: bool = False


class ScoreResponse(BaseModel):
    """Predictions and the immutable model version that produced them."""

    model_config = ConfigDict(extra="forbid")

    model: ModelResponseMetadata
    predictions: list[TransactionPrediction]


class ReadinessResponse(BaseModel):
    """Readiness state and the loaded registry version."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ready"]
    model_name: str
    model_version: str
    model_alias: str


ModelLoader = Callable[[str], Any]
MetadataResolver = Callable[
    [ScoringSettings],
    RegisteredModelMetadata,
]


def create_app(
    *,
    settings: ScoringSettings | None = None,
    model_loader: ModelLoader | None = None,
    metadata_resolver: MetadataResolver | None = None,
) -> FastAPI:
    """Create a scoring application with isolated runtime state."""
    resolved_settings = settings or ScoringSettings.from_environment()
    resolved_settings.validate()

    load_model = model_loader or mlflow.pyfunc.load_model
    resolve_metadata = metadata_resolver or resolve_registered_model_metadata

    prometheus_registry = CollectorRegistry()
    metrics = ScoringMetrics.create(prometheus_registry)

    @asynccontextmanager
    async def lifespan(
        application: FastAPI,
    ) -> Any:
        mlflow.set_tracking_uri(resolved_settings.tracking_uri)
        metadata = resolve_metadata(resolved_settings)
        model = load_model(metadata.model_uri)

        application.state.runtime = ScoringRuntime(
            model=model,
            metadata=metadata,
        )
        application.state.ready = True

        metrics.model_info.labels(
            model_name=metadata.model_name,
            model_version=metadata.model_version,
            model_alias=metadata.model_alias,
            source_revision=metadata.source_revision,
        ).set(1.0)

        LOGGER.info(
            "fraud_scoring_model_loaded model=%s version=%s alias=%s source_revision=%s",
            metadata.model_name,
            metadata.model_version,
            metadata.model_alias,
            metadata.source_revision,
        )

        try:
            yield
        finally:
            application.state.ready = False
            application.state.runtime = None

    application = FastAPI(
        title="Fraud Detection Scoring API",
        version="1.0.0",
        description=(
            "Score point-in-time transaction features with the approved MLflow fraud policy."
        ),
        lifespan=lifespan,
    )
    application.state.ready = False
    application.state.runtime = None
    application.state.prometheus_registry = prometheus_registry

    @application.middleware("http")
    async def observe_http_request(
        request: Request,
        call_next: Callable[..., Any],
    ) -> Response:
        endpoint = request.url.path if request.url.path in OBSERVED_ENDPOINTS else "other"

        if request.url.path == "/metrics":
            return await call_next(request)

        started_at = perf_counter()
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            metrics.requests_total.labels(
                endpoint=endpoint,
                method=request.method,
                status_code=str(status_code),
            ).inc()
            metrics.request_duration_seconds.labels(
                endpoint=endpoint,
                method=request.method,
            ).observe(perf_counter() - started_at)

    @application.get(
        "/health/live",
        tags=["health"],
    )
    def live() -> dict[str, str]:
        return {"status": "alive"}

    @application.get(
        "/health/ready",
        response_model=ReadinessResponse,
        tags=["health"],
    )
    def ready(
        request: Request,
    ) -> ReadinessResponse:
        if not request.app.state.ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Scoring model is not ready",
            )

        runtime = _runtime_from_request(request)

        return ReadinessResponse(
            status="ready",
            model_name=runtime.metadata.model_name,
            model_version=runtime.metadata.model_version,
            model_alias=runtime.metadata.model_alias,
        )

    @application.post(
        "/v1/score",
        response_model=ScoreResponse,
        tags=["scoring"],
    )
    def score(
        payload: ScoreRequest,
        request: Request,
    ) -> ScoreResponse:
        if len(payload.transactions) > resolved_settings.max_batch_size:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=(
                    f"Request exceeds the maximum batch size of {resolved_settings.max_batch_size}"
                ),
            )

        runtime = _runtime_from_request(request)
        model_input = _model_input_frame(payload)

        try:
            raw_predictions = runtime.model.predict(model_input)
            predictions = _validate_predictions(
                raw_predictions,
                expected_rows=len(payload.transactions),
            )
        except Exception as exc:
            LOGGER.exception("fraud_scoring_prediction_failed")
            raise HTTPException(
                status_code=(status.HTTP_500_INTERNAL_SERVER_ERROR),
                detail="Model prediction failed",
            ) from exc

        metrics.scored_transactions_total.inc(len(predictions))
        metrics.threshold_decisions_total.inc(int(predictions["threshold_decision"].sum()))

        for probability in predictions["fraud_probability"]:
            metrics.fraud_probability.observe(float(probability))

        response_predictions = [
            TransactionPrediction(
                transaction_id=transaction.transaction_id,
                customer_id=transaction.customer_id,
                terminal_id=transaction.terminal_id,
                fraud_probability=float(prediction.fraud_probability),
                threshold_decision=bool(prediction.threshold_decision),
            )
            for transaction, prediction in zip(
                payload.transactions,
                predictions.itertuples(index=False),
                strict=True,
            )
        ]

        return ScoreResponse(
            model=ModelResponseMetadata(
                name=runtime.metadata.model_name,
                version=runtime.metadata.model_version,
                alias=runtime.metadata.model_alias,
                source_revision=(runtime.metadata.source_revision),
                threshold=runtime.metadata.threshold,
                daily_card_capacity=(runtime.metadata.daily_card_capacity),
                capacity_scope=(runtime.metadata.capacity_scope),
                capacity_enforced_by_endpoint=False,
            ),
            predictions=response_predictions,
        )

    @application.get(
        "/metrics",
        include_in_schema=False,
    )
    def prometheus_metrics() -> Response:
        return Response(
            content=generate_latest(prometheus_registry),
            media_type=CONTENT_TYPE_LATEST,
        )

    return application


def resolve_registered_model_metadata(
    settings: ScoringSettings,
) -> RegisteredModelMetadata:
    """Resolve and validate the concrete approved alias version."""
    client = MlflowClient(tracking_uri=settings.tracking_uri)
    model_version = client.get_model_version_by_alias(
        settings.registered_model_name,
        settings.model_alias,
    )
    tags = model_version.tags or {}

    if tags.get("approval_status") != APPROVAL_STATUS:
        raise ValueError("Registered model alias must reference an approved model version")

    source_revision = _require_tag(
        tags,
        "source_revision",
    )
    threshold = _require_probability_tag(
        tags,
        "threshold",
    )
    daily_card_capacity = _require_positive_integer_tag(
        tags,
        "daily_card_capacity",
    )

    return RegisteredModelMetadata(
        model_name=settings.registered_model_name,
        model_version=str(model_version.version),
        model_alias=settings.model_alias,
        model_uri=(f"models:/{settings.registered_model_name}/{model_version.version}"),
        source_revision=source_revision,
        threshold=threshold,
        daily_card_capacity=daily_card_capacity,
        capacity_scope=CAPACITY_SCOPE,
    )


def _model_input_frame(
    payload: ScoreRequest,
) -> pd.DataFrame:
    records = [
        transaction.model_dump(include=set(MODEL_FEATURE_COLUMNS))
        for transaction in payload.transactions
    ]

    model_input = pd.DataFrame(
        records,
        columns=list(MODEL_FEATURE_COLUMNS),
    )

    return model_input.astype(MODEL_INPUT_DTYPES)


def _validate_predictions(
    predictions: Any,
    *,
    expected_rows: int,
) -> pd.DataFrame:
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("Model predictions must be a pandas DataFrame")

    required_columns = {
        "fraud_probability",
        "threshold_decision",
    }
    missing_columns = sorted(required_columns.difference(predictions.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Model predictions are missing columns: {missing}")

    if len(predictions) != expected_rows:
        raise ValueError("Model prediction count does not match the request")

    normalized = predictions.loc[
        :,
        [
            "fraud_probability",
            "threshold_decision",
        ],
    ].reset_index(drop=True)

    probability = pd.to_numeric(
        normalized["fraud_probability"],
        errors="coerce",
    )
    if probability.isna().any():
        raise ValueError("Model probabilities must be numeric")

    probability_values = probability.to_numpy(dtype=float)
    if not np.isfinite(probability_values).all():
        raise ValueError("Model probabilities must be finite")

    if ((probability_values < 0.0) | (probability_values > 1.0)).any():
        raise ValueError("Model probabilities must be between 0 and 1")

    threshold_decision = normalized["threshold_decision"]
    valid_decisions = threshold_decision.map(lambda value: isinstance(value, (bool, np.bool_)))
    if not valid_decisions.all():
        raise ValueError("Model threshold decisions must be boolean")

    normalized["fraud_probability"] = probability_values
    normalized["threshold_decision"] = threshold_decision.astype(bool)

    return normalized


def _runtime_from_request(
    request: Request,
) -> ScoringRuntime:
    runtime = request.app.state.runtime

    if not isinstance(runtime, ScoringRuntime):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scoring model is not ready",
        )

    return runtime


def _require_tag(
    tags: dict[str, str],
    key: str,
) -> str:
    value = tags.get(key)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Approved model version must contain tag: {key}")

    return value.strip()


def _require_probability_tag(
    tags: dict[str, str],
    key: str,
) -> float:
    value = _require_tag(tags, key)

    try:
        probability = float(value)
    except ValueError as exc:
        raise ValueError(f"Approved model tag must be numeric: {key}") from exc

    if not math.isfinite(probability) or not 0.0 < probability < 1.0:
        raise ValueError(f"Approved model tag must be between 0 and 1: {key}")

    return probability


def _require_positive_integer_tag(
    tags: dict[str, str],
    key: str,
) -> int:
    value = _require_tag(tags, key)

    try:
        integer_value = int(value)
    except ValueError as exc:
        raise ValueError(f"Approved model tag must be an integer: {key}") from exc

    if integer_value <= 0:
        raise ValueError(f"Approved model tag must be positive: {key}")

    return integer_value


app = create_app()
