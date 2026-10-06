import json
from datetime import UTC, date, datetime

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import types as T

from fraud_lakehouse.streaming_events import build_transaction_event
from fraud_lakehouse.streaming_transformations import (
    parse_kafka_transaction_events,
    project_transaction_events,
    select_quarantined_transaction_events,
    select_valid_transaction_events,
    type_transaction_events,
    validate_transaction_events,
)


def _transaction() -> dict[str, object]:
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


def _changed_value(
    event_value: bytes,
    *,
    transaction_updates: dict[str, object] | None = None,
    envelope_updates: dict[str, object] | None = None,
    removed_transaction_fields: tuple[str, ...] = (),
) -> bytes:
    payload = json.loads(event_value)

    if transaction_updates:
        payload["transaction"].update(transaction_updates)

    if envelope_updates:
        payload.update(envelope_updates)

    for field in removed_transaction_fields:
        payload["transaction"].pop(field)

    return json.dumps(
        payload,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def main() -> None:
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("streaming-quality-validation")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )

    try:
        event = build_transaction_event(
            transaction=_transaction(),
            source_file="2018-04-01.pkl",
            source_file_date=date(2018, 4, 1),
            source_row_number=0,
            produced_at_utc=datetime(
                2026,
                10,
                5,
                13,
                0,
                tzinfo=UTC,
            ),
        )

        kafka_schema = T.StructType(
            [
                T.StructField(
                    "key",
                    T.BinaryType(),
                    nullable=True,
                ),
                T.StructField(
                    "value",
                    T.BinaryType(),
                    nullable=False,
                ),
                T.StructField(
                    "topic",
                    T.StringType(),
                    nullable=False,
                ),
                T.StructField(
                    "partition",
                    T.IntegerType(),
                    nullable=False,
                ),
                T.StructField(
                    "offset",
                    T.LongType(),
                    nullable=False,
                ),
                T.StructField(
                    "timestamp",
                    T.TimestampType(),
                    nullable=False,
                ),
            ]
        )

        def record(
            offset: int,
            value: bytes,
            key: bytes | None = event.key,
        ) -> tuple[
            bytes | None,
            bytes,
            str,
            int,
            int,
            datetime,
        ]:
            return (
                key,
                value,
                "transactions.raw.v1",
                offset % 3,
                offset,
                datetime(
                    2026,
                    10,
                    5,
                    13,
                    0,
                    offset,
                ),
            )

        kafka_records = spark.createDataFrame(
            [
                record(0, event.value),
                record(
                    1,
                    _changed_value(
                        event.value,
                        transaction_updates={
                            "TX_AMOUNT": -1.0,
                            "TX_TIME_DAYS": 2,
                        },
                    ),
                ),
                record(
                    2,
                    _changed_value(
                        event.value,
                        transaction_updates={
                            "TRANSACTION_ID": "1001.5",
                        },
                    ),
                    key=b"1001.5",
                ),
                record(
                    3,
                    _changed_value(
                        event.value,
                        removed_transaction_fields=("TX_AMOUNT",),
                    ),
                ),
                record(
                    4,
                    _changed_value(
                        event.value,
                        transaction_updates={
                            "TX_FRAUD": 2,
                        },
                    ),
                ),
                record(
                    5,
                    _changed_value(
                        event.value,
                        transaction_updates={
                            "TX_FRAUD_SCENARIO": 2,
                        },
                    ),
                ),
                record(
                    6,
                    _changed_value(
                        event.value,
                        envelope_updates={
                            "source_file_date": ("2018-04-02"),
                        },
                    ),
                ),
                record(
                    7,
                    b"{not-valid-json",
                    key=b"malformed",
                ),
                record(
                    8,
                    _changed_value(
                        event.value,
                        envelope_updates={
                            "schema_version": "2.0",
                        },
                    ),
                ),
                record(
                    9,
                    event.value,
                    key=b"9999",
                ),
                record(
                    10,
                    _changed_value(
                        event.value,
                        envelope_updates={
                            "event_id": "not-a-uuid",
                        },
                    ),
                ),
                record(
                    11,
                    _changed_value(
                        event.value,
                        envelope_updates={
                            "source_file": None,
                        },
                    ),
                ),
            ],
            schema=kafka_schema,
        )

        validated = validate_transaction_events(
            type_transaction_events(
                project_transaction_events(parse_kafka_transaction_events(kafka_records))
            )
        )

        actual_reasons = {
            row["kafka_offset"]: row["rejection_reasons"]
            for row in validated.select(
                "kafka_offset",
                "rejection_reasons",
            ).collect()
        }

        assert actual_reasons == {
            0: [],
            1: [
                "negative_transaction_amount",
                "inconsistent_time_fields",
            ],
            2: ["invalid_data_type"],
            3: ["missing_required_value"],
            4: ["invalid_fraud_label"],
            5: ["invalid_fraud_scenario"],
            6: ["source_date_mismatch"],
            7: ["invalid_event_json"],
            8: ["unsupported_schema_version"],
            9: ["kafka_key_mismatch"],
            10: ["invalid_event_metadata"],
            11: ["missing_event_metadata"],
        }

        valid_offsets = [
            row["kafka_offset"]
            for row in select_valid_transaction_events(validated).select("kafka_offset").collect()
        ]
        quarantine_offsets = sorted(
            row["kafka_offset"]
            for row in (
                select_quarantined_transaction_events(validated).select("kafka_offset").collect()
            )
        )

        assert valid_offsets == [0]
        assert quarantine_offsets == list(range(1, 12))

        print("Spark streaming quality validation passed")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
