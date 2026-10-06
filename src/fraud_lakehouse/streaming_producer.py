import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Protocol

from confluent_kafka import KafkaException, Producer

from fraud_lakehouse.bronze import (
    SOURCE_COLUMNS,
    discover_source_files,
    load_trusted_source_file,
)
from fraud_lakehouse.streaming_events import (
    TRANSACTION_TOPIC,
    SerializedKafkaRecord,
    build_transaction_event,
)

_DEFAULT_CLIENT_ID = "fraud-lakehouse-replay-producer"
_FLUSH_TIMEOUT_SECONDS = 30.0


class ProducerProtocol(Protocol):
    """Kafka producer operations used by the replay workflow."""

    def produce(
        self,
        topic: str,
        *,
        key: bytes | None,
        value: bytes,
        on_delivery: Callable[[object | None, object], None],
    ) -> None: ...

    def poll(self, timeout: float) -> int: ...

    def flush(self, timeout: float) -> int: ...


ProducerFactory = Callable[[dict[str, object]], ProducerProtocol]


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """Summary of a completed transaction replay."""

    source_files: tuple[Path, ...]
    events_produced: int


def _create_producer(
    configuration: dict[str, object],
) -> ProducerProtocol:
    return Producer(configuration)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _validate_replay_options(
    *,
    bootstrap_servers: str,
    topic: str,
    events_per_second: float,
    max_events: int | None,
) -> None:
    if not bootstrap_servers.strip():
        raise ValueError("bootstrap_servers must not be empty")

    if not topic.strip():
        raise ValueError("topic must not be empty")

    if events_per_second < 0:
        raise ValueError("events_per_second must be greater than or equal to zero")

    if max_events is not None and max_events <= 0:
        raise ValueError("max_events must be greater than zero")


def _produce_record(
    producer: ProducerProtocol,
    *,
    topic: str,
    record: SerializedKafkaRecord,
    on_delivery: Callable[[object | None, object], None],
) -> None:
    while True:
        try:
            producer.produce(
                topic,
                key=record.key,
                value=record.value,
                on_delivery=on_delivery,
            )
            producer.poll(0)
            return
        except BufferError:
            producer.poll(1.0)


def replay_transactions(
    *,
    raw_dir: Path,
    bootstrap_servers: str,
    topic: str = TRANSACTION_TOPIC,
    events_per_second: float = 100.0,
    max_events: int | None = None,
    producer_factory: ProducerFactory = _create_producer,
    clock: Callable[[], datetime] = _utc_now,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> ReplayResult:
    """Replay trusted daily transaction files into a Kafka topic."""
    _validate_replay_options(
        bootstrap_servers=bootstrap_servers,
        topic=topic,
        events_per_second=events_per_second,
        max_events=max_events,
    )

    configuration: dict[str, object] = {
        "bootstrap.servers": bootstrap_servers,
        "client.id": _DEFAULT_CLIENT_ID,
        "enable.idempotence": True,
        "acks": "all",
    }

    try:
        producer = producer_factory(configuration)
    except KafkaException as error:
        raise RuntimeError("Failed to create Kafka producer") from error

    delivery_errors: list[str] = []

    def record_delivery(
        error: object | None,
        _message: object,
    ) -> None:
        if error is not None:
            delivery_errors.append(str(error))

    interval_seconds = 0.0 if events_per_second == 0 else 1.0 / events_per_second
    next_event_at = monotonic()
    events_produced = 0
    processed_files: list[Path] = []

    try:
        for source_file in discover_source_files(raw_dir):
            source_data = load_trusted_source_file(source_file)
            source_file_date = date.fromisoformat(source_file.stem)
            processed_files.append(source_file)

            source_rows = source_data.loc[
                :,
                list(SOURCE_COLUMNS),
            ].itertuples(index=False, name=None)

            for source_row_number, values in enumerate(source_rows):
                if max_events is not None and events_produced >= max_events:
                    break

                if interval_seconds > 0 and events_produced > 0:
                    next_event_at += interval_seconds
                    delay = next_event_at - monotonic()
                    if delay > 0:
                        sleep(delay)

                transaction = dict(
                    zip(
                        SOURCE_COLUMNS,
                        values,
                        strict=True,
                    )
                )

                record = build_transaction_event(
                    transaction=transaction,
                    source_file=source_file.name,
                    source_file_date=source_file_date,
                    source_row_number=source_row_number,
                    produced_at_utc=clock(),
                )

                _produce_record(
                    producer,
                    topic=topic,
                    record=record,
                    on_delivery=record_delivery,
                )
                events_produced += 1

            if max_events is not None and events_produced >= max_events:
                break

        remaining_messages = producer.flush(_FLUSH_TIMEOUT_SECONDS)
    except KafkaException as error:
        raise RuntimeError("Kafka transaction replay failed") from error

    if remaining_messages:
        raise RuntimeError(f"Kafka producer did not deliver {remaining_messages} queued message(s)")

    if delivery_errors:
        details = "; ".join(delivery_errors)
        raise RuntimeError(f"Kafka delivery failed: {details}")

    return ReplayResult(
        source_files=tuple(processed_files),
        events_produced=events_produced,
    )
