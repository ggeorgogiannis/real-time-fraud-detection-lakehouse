import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from uuid import NAMESPACE_URL, uuid5

import pandas as pd

from fraud_lakehouse.bronze import SOURCE_COLUMNS

TRANSACTION_EVENT_SCHEMA_VERSION = "1.0"
TRANSACTION_TOPIC = "transactions.raw.v1"


@dataclass(frozen=True, slots=True)
class SerializedKafkaRecord:
    """Serialized Kafka key and event value."""

    key: bytes | None
    value: bytes


def _validate_utc(timestamp: datetime) -> None:
    if timestamp.tzinfo is None or timestamp.utcoffset() != timedelta(0):
        raise ValueError("produced_at_utc must be timezone-aware UTC")


def _format_utc(timestamp: datetime) -> str:
    return timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _normalize_source_value(value: object) -> object:
    """Convert a source scalar into a standards-compliant JSON value."""
    if value is None:
        return None

    missing = pd.isna(value)
    if not hasattr(missing, "__len__") and bool(missing):
        return None

    if isinstance(value, pd.Timestamp):
        timestamp = value.to_pydatetime()
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        return _format_utc(timestamp)

    if isinstance(value, datetime):
        timestamp = value
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        return _format_utc(timestamp)

    if isinstance(value, date):
        return value.isoformat()

    item_method = getattr(value, "item", None)
    if callable(item_method):
        value = item_method()

    if isinstance(value, float) and not math.isfinite(value):
        return str(value)

    if isinstance(value, (str, int, float, bool)):
        return value

    raise TypeError(f"Unsupported transaction value type: {type(value).__name__}")


def _validate_transaction_schema(
    transaction: Mapping[str, object],
) -> None:
    required_columns = set(SOURCE_COLUMNS)
    actual_columns = set(transaction)

    missing_columns = sorted(required_columns - actual_columns)
    if missing_columns:
        details = ", ".join(missing_columns)
        raise ValueError(f"Transaction is missing required columns: {details}")

    unexpected_columns = sorted(actual_columns - required_columns)
    if unexpected_columns:
        details = ", ".join(unexpected_columns)
        raise ValueError(f"Transaction contains unexpected columns: {details}")


def _build_event_id(
    source_file: str,
    source_row_number: int,
) -> str:
    identity = f"fraud-lakehouse://transactions/{source_file}#{source_row_number}"
    return str(uuid5(NAMESPACE_URL, identity))


def _build_kafka_key(transaction_id: object) -> bytes | None:
    normalized = _normalize_source_value(transaction_id)

    if normalized is None:
        return None

    if isinstance(normalized, bool):
        return str(normalized).lower().encode("utf-8")

    if isinstance(normalized, int):
        return str(normalized).encode("utf-8")

    if isinstance(normalized, float) and math.isfinite(normalized) and normalized.is_integer():
        return str(int(normalized)).encode("utf-8")

    return str(normalized).encode("utf-8")


def build_transaction_event(
    *,
    transaction: Mapping[str, object],
    source_file: str,
    source_file_date: date,
    source_row_number: int,
    produced_at_utc: datetime,
) -> SerializedKafkaRecord:
    """Build a deterministic, versioned Kafka transaction event."""
    _validate_utc(produced_at_utc)
    _validate_transaction_schema(transaction)

    if not source_file:
        raise ValueError("source_file must not be empty")

    if isinstance(source_row_number, bool) or source_row_number < 0:
        raise ValueError("source_row_number must be a non-negative integer")

    if not isinstance(source_row_number, int):
        raise TypeError("source_row_number must be an integer")

    normalized_transaction = {
        column: _normalize_source_value(transaction[column]) for column in SOURCE_COLUMNS
    }

    event = {
        "schema_version": TRANSACTION_EVENT_SCHEMA_VERSION,
        "event_id": _build_event_id(
            source_file,
            source_row_number,
        ),
        "produced_at_utc": _format_utc(produced_at_utc),
        "source_file": source_file,
        "source_file_date": source_file_date.isoformat(),
        "source_row_number": source_row_number,
        "transaction": normalized_transaction,
    }

    serialized_value = json.dumps(
        event,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    return SerializedKafkaRecord(
        key=_build_kafka_key(transaction["TRANSACTION_ID"]),
        value=serialized_value,
    )
