from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

from fraud_lakehouse.bronze import SOURCE_COLUMNS


def transaction_event_schema() -> T.StructType:
    """Return the explicit schema for version 1.0 transaction events."""
    transaction_schema = T.StructType(
        [
            T.StructField(
                column,
                T.StringType(),
                nullable=True,
            )
            for column in SOURCE_COLUMNS
        ]
    )

    return T.StructType(
        [
            T.StructField(
                "schema_version",
                T.StringType(),
                nullable=True,
            ),
            T.StructField(
                "event_id",
                T.StringType(),
                nullable=True,
            ),
            T.StructField(
                "produced_at_utc",
                T.StringType(),
                nullable=True,
            ),
            T.StructField(
                "source_file",
                T.StringType(),
                nullable=True,
            ),
            T.StructField(
                "source_file_date",
                T.StringType(),
                nullable=True,
            ),
            T.StructField(
                "source_row_number",
                T.LongType(),
                nullable=True,
            ),
            T.StructField(
                "transaction",
                transaction_schema,
                nullable=True,
            ),
        ]
    )


def parse_kafka_transaction_events(
    kafka_records: DataFrame,
) -> DataFrame:
    """Preserve Kafka metadata and parse raw event JSON."""
    bronze = kafka_records.select(
        F.col("key").cast("string").alias("kafka_key"),
        F.col("value").cast("string").alias("raw_event"),
        F.col("topic").alias("kafka_topic"),
        F.col("partition").alias("kafka_partition"),
        F.col("offset").alias("kafka_offset"),
        F.col("timestamp").alias("kafka_timestamp"),
    )

    return bronze.withColumn(
        "event",
        F.from_json(
            F.col("raw_event"),
            transaction_event_schema(),
        ),
    )


def project_transaction_events(
    parsed_events: DataFrame,
) -> DataFrame:
    """Flatten parsed envelope and transaction fields."""
    transaction_columns = [
        F.col(f"event.transaction.`{column}`").alias(column) for column in SOURCE_COLUMNS
    ]

    return parsed_events.select(
        "kafka_key",
        "raw_event",
        "kafka_topic",
        "kafka_partition",
        "kafka_offset",
        "kafka_timestamp",
        F.col("event.schema_version").alias("schema_version"),
        F.col("event.event_id").alias("event_id"),
        F.col("event.produced_at_utc").alias("produced_at_utc"),
        F.col("event.source_file").alias("source_file"),
        F.col("event.source_file_date").alias("source_file_date"),
        F.col("event.source_row_number").alias("source_row_number"),
        *transaction_columns,
    )
