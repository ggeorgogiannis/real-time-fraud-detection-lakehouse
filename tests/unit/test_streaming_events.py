import json
from datetime import UTC, date, datetime, timedelta, timezone
from uuid import UUID

import pandas as pd
import pytest

from fraud_lakehouse.streaming_events import (
    TRANSACTION_EVENT_SCHEMA_VERSION,
    build_transaction_event,
)


def _valid_transaction() -> dict[str, object]:
    return {
        "TRANSACTION_ID": 1001,
        "TX_DATETIME": pd.Timestamp("2018-04-01 08:00:00"),
        "CUSTOMER_ID": 101,
        "TERMINAL_ID": 201,
        "TX_AMOUNT": 25.5,
        "TX_TIME_SECONDS": 28800,
        "TX_TIME_DAYS": 0,
        "TX_FRAUD": 0,
        "TX_FRAUD_SCENARIO": 0,
    }


def test_build_transaction_event_serializes_contract() -> None:
    produced_at = datetime(2026, 10, 5, 13, 0, tzinfo=UTC)

    record = build_transaction_event(
        transaction=_valid_transaction(),
        source_file="2018-04-01.pkl",
        source_file_date=date(2018, 4, 1),
        source_row_number=0,
        produced_at_utc=produced_at,
    )

    event = json.loads(record.value)

    assert record.key == b"1001"
    assert event["schema_version"] == TRANSACTION_EVENT_SCHEMA_VERSION
    assert UUID(event["event_id"]).version == 5
    assert event["produced_at_utc"] == "2026-10-05T13:00:00Z"
    assert event["source_file"] == "2018-04-01.pkl"
    assert event["source_file_date"] == "2018-04-01"
    assert event["source_row_number"] == 0
    assert event["transaction"]["TRANSACTION_ID"] == 1001
    assert event["transaction"]["TX_DATETIME"] == "2018-04-01T08:00:00Z"
    assert event["transaction"]["TX_AMOUNT"] == 25.5


def test_event_id_is_deterministic_for_source_position() -> None:
    first = build_transaction_event(
        transaction=_valid_transaction(),
        source_file="2018-04-01.pkl",
        source_file_date=date(2018, 4, 1),
        source_row_number=4,
        produced_at_utc=datetime(2026, 10, 5, 13, 0, tzinfo=UTC),
    )
    second = build_transaction_event(
        transaction=_valid_transaction(),
        source_file="2018-04-01.pkl",
        source_file_date=date(2018, 4, 1),
        source_row_number=4,
        produced_at_utc=datetime(2026, 10, 5, 14, 0, tzinfo=UTC),
    )

    assert json.loads(first.value)["event_id"] == json.loads(second.value)["event_id"]


def test_event_id_changes_for_different_source_position() -> None:
    first = build_transaction_event(
        transaction=_valid_transaction(),
        source_file="2018-04-01.pkl",
        source_file_date=date(2018, 4, 1),
        source_row_number=0,
        produced_at_utc=datetime(2026, 10, 5, 13, 0, tzinfo=UTC),
    )
    second = build_transaction_event(
        transaction=_valid_transaction(),
        source_file="2018-04-01.pkl",
        source_file_date=date(2018, 4, 1),
        source_row_number=1,
        produced_at_utc=datetime(2026, 10, 5, 13, 0, tzinfo=UTC),
    )

    assert json.loads(first.value)["event_id"] != json.loads(second.value)["event_id"]


def test_build_transaction_event_preserves_invalid_values() -> None:
    transaction = _valid_transaction()
    transaction["CUSTOMER_ID"] = "not-a-number"
    transaction["TX_AMOUNT"] = pd.NA
    transaction["TX_TIME_SECONDS"] = float("inf")

    record = build_transaction_event(
        transaction=transaction,
        source_file="2018-04-01.pkl",
        source_file_date=date(2018, 4, 1),
        source_row_number=0,
        produced_at_utc=datetime(2026, 10, 5, 13, 0, tzinfo=UTC),
    )

    payload = json.loads(record.value)["transaction"]

    assert payload["CUSTOMER_ID"] == "not-a-number"
    assert payload["TX_AMOUNT"] is None
    assert payload["TX_TIME_SECONDS"] == "inf"


def test_missing_transaction_id_produces_null_kafka_key() -> None:
    transaction = _valid_transaction()
    transaction["TRANSACTION_ID"] = pd.NA

    record = build_transaction_event(
        transaction=transaction,
        source_file="2018-04-01.pkl",
        source_file_date=date(2018, 4, 1),
        source_row_number=0,
        produced_at_utc=datetime(2026, 10, 5, 13, 0, tzinfo=UTC),
    )

    assert record.key is None


def test_build_transaction_event_rejects_missing_column() -> None:
    transaction = _valid_transaction()
    del transaction["TX_AMOUNT"]

    with pytest.raises(
        ValueError,
        match="missing required columns: TX_AMOUNT",
    ):
        build_transaction_event(
            transaction=transaction,
            source_file="2018-04-01.pkl",
            source_file_date=date(2018, 4, 1),
            source_row_number=0,
            produced_at_utc=datetime(2026, 10, 5, 13, 0, tzinfo=UTC),
        )


def test_build_transaction_event_rejects_schema_drift() -> None:
    transaction = _valid_transaction()
    transaction["UNEXPECTED"] = "value"

    with pytest.raises(
        ValueError,
        match="unexpected columns: UNEXPECTED",
    ):
        build_transaction_event(
            transaction=transaction,
            source_file="2018-04-01.pkl",
            source_file_date=date(2018, 4, 1),
            source_row_number=0,
            produced_at_utc=datetime(2026, 10, 5, 13, 0, tzinfo=UTC),
        )


def test_build_transaction_event_requires_utc_timestamp() -> None:
    non_utc = datetime(
        2026,
        10,
        5,
        15,
        0,
        tzinfo=timezone(timedelta(hours=2)),
    )

    with pytest.raises(
        ValueError,
        match="produced_at_utc must be timezone-aware UTC",
    ):
        build_transaction_event(
            transaction=_valid_transaction(),
            source_file="2018-04-01.pkl",
            source_file_date=date(2018, 4, 1),
            source_row_number=0,
            produced_at_utc=non_utc,
        )


@pytest.mark.parametrize("source_row_number", [-1, True])
def test_build_transaction_event_rejects_invalid_source_row_number(
    source_row_number: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="source_row_number must be a non-negative integer",
    ):
        build_transaction_event(
            transaction=_valid_transaction(),
            source_file="2018-04-01.pkl",
            source_file_date=date(2018, 4, 1),
            source_row_number=source_row_number,
            produced_at_utc=datetime(2026, 10, 5, 13, 0, tzinfo=UTC),
        )
