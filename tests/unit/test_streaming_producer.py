import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from fraud_lakehouse.streaming_producer import replay_transactions

FIXTURES_DIR = Path(__file__).parents[1] / "fixtures"


class FakeProducer:
    def __init__(
        self,
        configuration: dict[str, object],
        *,
        delivery_error: object | None = None,
        remaining_messages: int = 0,
    ) -> None:
        self.configuration = configuration
        self.delivery_error = delivery_error
        self.remaining_messages = remaining_messages
        self.records: list[dict[str, object]] = []
        self.flush_timeout: float | None = None

    def produce(
        self,
        topic: str,
        *,
        key: bytes | None,
        value: bytes,
        on_delivery: Callable[[object | None, object], None],
    ) -> None:
        self.records.append(
            {
                "topic": topic,
                "key": key,
                "value": value,
            }
        )
        on_delivery(self.delivery_error, object())

    def poll(self, timeout: float) -> int:
        return 0

    def flush(self, timeout: float) -> int:
        self.flush_timeout = timeout
        return self.remaining_messages


def _write_source_file(
    raw_dir: Path,
    *,
    fixture_name: str = "transactions_valid.csv",
    source_name: str = "2018-04-01.pkl",
) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    source_data = pd.read_csv(FIXTURES_DIR / fixture_name)
    source_path = raw_dir / source_name
    source_data.to_pickle(source_path)
    return source_path


def test_replay_transactions_produces_versioned_events(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    source_path = _write_source_file(raw_dir)
    fixed_time = datetime(2026, 10, 5, 13, 0, tzinfo=UTC)
    created_producers: list[FakeProducer] = []

    def producer_factory(
        configuration: dict[str, object],
    ) -> FakeProducer:
        producer = FakeProducer(configuration)
        created_producers.append(producer)
        return producer

    result = replay_transactions(
        raw_dir=raw_dir,
        bootstrap_servers="localhost:9092",
        events_per_second=0,
        producer_factory=producer_factory,
        clock=lambda: fixed_time,
    )

    assert result.source_files == (source_path,)
    assert result.events_produced == 5
    assert len(created_producers) == 1

    producer = created_producers[0]
    assert producer.configuration == {
        "bootstrap.servers": "localhost:9092",
        "client.id": "fraud-lakehouse-replay-producer",
        "enable.idempotence": True,
        "acks": "all",
    }
    assert producer.flush_timeout == 30.0
    assert len(producer.records) == 5

    first_record = producer.records[0]
    first_event = json.loads(first_record["value"])

    assert first_record["topic"] == "transactions.raw.v1"
    assert first_record["key"] == b"1001"
    assert first_event["schema_version"] == "1.0"
    assert first_event["source_file"] == "2018-04-01.pkl"
    assert first_event["source_row_number"] == 0
    assert first_event["transaction"]["TRANSACTION_ID"] == 1001


def test_replay_transactions_respects_event_limit(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    _write_source_file(raw_dir)
    producer = FakeProducer({})

    result = replay_transactions(
        raw_dir=raw_dir,
        bootstrap_servers="localhost:9092",
        events_per_second=0,
        max_events=2,
        producer_factory=lambda _configuration: producer,
        clock=lambda: datetime(2026, 10, 5, 13, 0, tzinfo=UTC),
    )

    assert result.events_produced == 2
    assert len(producer.records) == 2


def test_replay_transactions_preserves_invalid_records(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    _write_source_file(
        raw_dir,
        fixture_name="transactions_invalid.csv",
    )
    producer = FakeProducer({})

    result = replay_transactions(
        raw_dir=raw_dir,
        bootstrap_servers="localhost:9092",
        events_per_second=0,
        producer_factory=lambda _configuration: producer,
        clock=lambda: datetime(2026, 10, 5, 13, 0, tzinfo=UTC),
    )

    events = [json.loads(record["value"]) for record in producer.records]

    assert result.events_produced == 7
    assert events[0]["transaction"]["TX_AMOUNT"] is None
    assert events[5]["transaction"]["CUSTOMER_ID"] == "not-a-number"


@pytest.mark.parametrize(
    ("arguments", "expected_message"),
    [
        (
            {"bootstrap_servers": ""},
            "bootstrap_servers must not be empty",
        ),
        (
            {"topic": ""},
            "topic must not be empty",
        ),
        (
            {"events_per_second": -1.0},
            "events_per_second must be greater than or equal to zero",
        ),
        (
            {"max_events": 0},
            "max_events must be greater than zero",
        ),
    ],
)
def test_replay_transactions_rejects_invalid_options(
    tmp_path: Path,
    arguments: dict[str, object],
    expected_message: str,
) -> None:
    options: dict[str, object] = {
        "raw_dir": tmp_path,
        "bootstrap_servers": "localhost:9092",
        "events_per_second": 0,
    }
    options.update(arguments)

    with pytest.raises(ValueError, match=expected_message):
        replay_transactions(**options)


def test_replay_transactions_reports_delivery_failure(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    _write_source_file(raw_dir)
    producer = FakeProducer(
        {},
        delivery_error="broker unavailable",
    )

    with pytest.raises(
        RuntimeError,
        match="Kafka delivery failed: broker unavailable",
    ):
        replay_transactions(
            raw_dir=raw_dir,
            bootstrap_servers="localhost:9092",
            events_per_second=0,
            producer_factory=lambda _configuration: producer,
            clock=lambda: datetime(2026, 10, 5, 13, 0, tzinfo=UTC),
        )


def test_replay_transactions_reports_undelivered_messages(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    _write_source_file(raw_dir)
    producer = FakeProducer(
        {},
        remaining_messages=2,
    )

    with pytest.raises(
        RuntimeError,
        match="did not deliver 2 queued message",
    ):
        replay_transactions(
            raw_dir=raw_dir,
            bootstrap_servers="localhost:9092",
            events_per_second=0,
            producer_factory=lambda _configuration: producer,
            clock=lambda: datetime(2026, 10, 5, 13, 0, tzinfo=UTC),
        )
