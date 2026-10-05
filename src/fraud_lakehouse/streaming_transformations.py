from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

from fraud_lakehouse.bronze import SOURCE_COLUMNS
from fraud_lakehouse.streaming_events import (
    TRANSACTION_EVENT_SCHEMA_VERSION,
)

INTEGER_PATTERN = r"^[+-]?[0-9]+$"

DEFAULT_TRANSACTION_WATERMARK_DELAY = "1 day"

RAW_TRANSACTION_COLUMNS = {column: f"raw_{column.lower()}" for column in SOURCE_COLUMNS}

UUID_PATTERN = (
    r"^[0-9a-fA-F]{8}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12}$"
)


def _any_true(*conditions: Column) -> Column:
    result = F.lit(False)

    for condition in conditions:
        result = result | F.coalesce(
            condition,
            F.lit(False),
        )

    return result


def validate_transaction_events(
    typed_events: DataFrame,
) -> DataFrame:
    """Attach deterministic rejection reasons to typed events."""
    event_json_is_valid = F.get_json_object(
        F.col("raw_event"),
        "$",
    ).isNotNull()

    missing_event_metadata = _any_true(
        F.col("kafka_key").isNull(),
        F.col("schema_version").isNull(),
        F.col("event_id").isNull(),
        F.col("produced_at_utc").isNull(),
        F.col("source_file").isNull(),
        F.col("source_file_date").isNull(),
        F.col("source_row_number").isNull(),
    )

    invalid_event_metadata = _any_true(
        F.col("event_id").isNotNull() & ~F.col("event_id").rlike(UUID_PATTERN),
        F.col("produced_at_utc").isNotNull() & F.col("produced_at").isNull(),
        F.col("source_file_date").isNotNull() & F.col("source_date").isNull(),
        F.col("source_row_number").isNotNull() & (F.col("source_row_number") < 0),
    )

    unsupported_schema_version = F.col("schema_version").isNotNull() & (
        F.col("schema_version") != F.lit(TRANSACTION_EVENT_SCHEMA_VERSION)
    )

    missing_required_value = _any_true(
        *[F.col(raw_column).isNull() for raw_column in RAW_TRANSACTION_COLUMNS.values()]
    )

    invalid_data_type = _any_true(
        F.col("raw_transaction_id").isNotNull() & F.col("transaction_id").isNull(),
        F.col("raw_tx_datetime").isNotNull() & F.col("tx_datetime").isNull(),
        F.col("raw_customer_id").isNotNull() & F.col("customer_id").isNull(),
        F.col("raw_terminal_id").isNotNull() & F.col("terminal_id").isNull(),
        F.col("raw_tx_amount").isNotNull()
        & (
            F.col("tx_amount").isNull()
            | F.isnan("tx_amount")
            | F.col("tx_amount").isin(
                float("inf"),
                float("-inf"),
            )
        ),
        F.col("raw_tx_time_seconds").isNotNull() & F.col("tx_time_seconds").isNull(),
        F.col("raw_tx_time_days").isNotNull() & F.col("tx_time_days").isNull(),
        F.col("raw_tx_fraud").isNotNull() & F.col("tx_fraud").isNull(),
        F.col("raw_tx_fraud_scenario").isNotNull() & F.col("tx_fraud_scenario").isNull(),
        F.col("transaction_id").isNotNull() & (F.col("transaction_id") < 0),
        F.col("customer_id").isNotNull() & (F.col("customer_id") < 0),
        F.col("terminal_id").isNotNull() & (F.col("terminal_id") < 0),
    )

    negative_transaction_amount = (
        F.col("tx_amount").isNotNull() & ~F.isnan("tx_amount") & (F.col("tx_amount") < 0)
    )

    valid_fraud_label = F.col("tx_fraud").isNotNull() & F.col("tx_fraud").isin(0, 1)
    invalid_fraud_label = F.col("tx_fraud").isNotNull() & ~F.col("tx_fraud").isin(0, 1)

    valid_fraud_scenario = F.col("tx_fraud_scenario").isNotNull() & F.col("tx_fraud_scenario").isin(
        0, 1, 2, 3
    )
    inconsistent_fraud_fields = (
        valid_fraud_label
        & valid_fraud_scenario
        & (
            ((F.col("tx_fraud_scenario") == 0) & (F.col("tx_fraud") != 0))
            | (F.col("tx_fraud_scenario").isin(1, 2, 3) & (F.col("tx_fraud") != 1))
        )
    )
    invalid_fraud_scenario = (
        F.col("tx_fraud_scenario").isNotNull()
        & ~F.col("tx_fraud_scenario").isin(
            0,
            1,
            2,
            3,
        )
    ) | inconsistent_fraud_fields

    valid_time_fields = F.col("tx_time_seconds").isNotNull() & F.col("tx_time_days").isNotNull()
    inconsistent_time_fields = valid_time_fields & (
        (F.col("tx_time_seconds") < 0)
        | (F.col("tx_time_days") < 0)
        | (F.col("tx_time_days") != F.floor(F.col("tx_time_seconds") / F.lit(86400)))
    )

    source_date_mismatch = (
        F.col("tx_datetime").isNotNull()
        & F.col("source_date").isNotNull()
        & (F.to_date("tx_datetime") != F.col("source_date"))
    )

    kafka_key_mismatch = (
        F.col("kafka_key").isNotNull()
        & F.col("transaction_id").isNotNull()
        & (F.col("kafka_key") != F.col("transaction_id").cast("string"))
    )

    rejection_reasons = F.array_compact(
        F.array(
            F.when(
                ~event_json_is_valid,
                F.lit("invalid_event_json"),
            ),
            F.when(
                event_json_is_valid & missing_event_metadata,
                F.lit("missing_event_metadata"),
            ),
            F.when(
                event_json_is_valid & invalid_event_metadata,
                F.lit("invalid_event_metadata"),
            ),
            F.when(
                event_json_is_valid & unsupported_schema_version,
                F.lit("unsupported_schema_version"),
            ),
            F.when(
                event_json_is_valid & missing_required_value,
                F.lit("missing_required_value"),
            ),
            F.when(
                event_json_is_valid & invalid_data_type,
                F.lit("invalid_data_type"),
            ),
            F.when(
                event_json_is_valid & negative_transaction_amount,
                F.lit("negative_transaction_amount"),
            ),
            F.when(
                event_json_is_valid & invalid_fraud_label,
                F.lit("invalid_fraud_label"),
            ),
            F.when(
                event_json_is_valid & invalid_fraud_scenario,
                F.lit("invalid_fraud_scenario"),
            ),
            F.when(
                event_json_is_valid & inconsistent_time_fields,
                F.lit("inconsistent_time_fields"),
            ),
            F.when(
                event_json_is_valid & source_date_mismatch,
                F.lit("source_date_mismatch"),
            ),
            F.when(
                event_json_is_valid & kafka_key_mismatch,
                F.lit("kafka_key_mismatch"),
            ),
        )
    )

    return typed_events.withColumn(
        "rejection_reasons",
        rejection_reasons,
    ).withColumn(
        "is_valid",
        F.size("rejection_reasons") == 0,
    )


def select_valid_transaction_events(
    validated_events: DataFrame,
) -> DataFrame:
    """Return events accepted by the streaming data contract."""
    return validated_events.filter(F.col("is_valid"))


def deduplicate_transaction_events(
    valid_events: DataFrame,
    watermark_delay: str = DEFAULT_TRANSACTION_WATERMARK_DELAY,
) -> DataFrame:
    """Deduplicate valid transactions using bounded event-time state."""
    if not watermark_delay.strip():
        raise ValueError("watermark_delay must not be empty")

    return valid_events.withWatermark(
        "tx_datetime",
        watermark_delay,
    ).dropDuplicatesWithinWatermark(
        ["transaction_id"],
    )


def select_quarantined_transaction_events(
    validated_events: DataFrame,
) -> DataFrame:
    """Return events rejected by the streaming data contract."""
    return validated_events.filter(~F.col("is_valid"))


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


def _try_parse_integer(
    column: str,
    data_type: str,
) -> Column:
    value = F.trim(F.col(column))

    return F.when(
        value.rlike(INTEGER_PATTERN),
        value.try_cast(data_type),
    )


def type_transaction_events(
    projected_events: DataFrame,
) -> DataFrame:
    """Preserve raw values and create canonical typed fields."""
    typed_events = projected_events

    for source_column, raw_column in RAW_TRANSACTION_COLUMNS.items():
        typed_events = typed_events.withColumnRenamed(
            source_column,
            raw_column,
        )

    return (
        typed_events.withColumn(
            "produced_at",
            F.try_to_timestamp("produced_at_utc"),
        )
        .withColumn(
            "source_date",
            F.col("source_file_date").try_cast("date"),
        )
        .withColumn(
            "transaction_id",
            _try_parse_integer(
                "raw_transaction_id",
                "long",
            ),
        )
        .withColumn(
            "tx_datetime",
            F.try_to_timestamp("raw_tx_datetime"),
        )
        .withColumn(
            "customer_id",
            _try_parse_integer(
                "raw_customer_id",
                "long",
            ),
        )
        .withColumn(
            "terminal_id",
            _try_parse_integer(
                "raw_terminal_id",
                "long",
            ),
        )
        .withColumn(
            "tx_amount",
            F.col("raw_tx_amount").try_cast("double"),
        )
        .withColumn(
            "tx_time_seconds",
            _try_parse_integer(
                "raw_tx_time_seconds",
                "long",
            ),
        )
        .withColumn(
            "tx_time_days",
            _try_parse_integer(
                "raw_tx_time_days",
                "long",
            ),
        )
        .withColumn(
            "tx_fraud",
            _try_parse_integer(
                "raw_tx_fraud",
                "byte",
            ),
        )
        .withColumn(
            "tx_fraud_scenario",
            _try_parse_integer(
                "raw_tx_fraud_scenario",
                "byte",
            ),
        )
    )


def project_streaming_bronze(
    parsed_events: DataFrame,
) -> DataFrame:
    """Create the immutable raw streaming Bronze representation."""
    return parsed_events.select(
        "kafka_key",
        "raw_event",
        "kafka_topic",
        "kafka_partition",
        "kafka_offset",
        "kafka_timestamp",
        F.col("kafka_timestamp").alias("ingested_at_utc"),
    )


def project_streaming_silver(
    deduplicated_events: DataFrame,
) -> DataFrame:
    """Create canonical Silver transactions with streaming metadata."""
    return deduplicated_events.select(
        "transaction_id",
        "tx_datetime",
        "customer_id",
        "terminal_id",
        "tx_amount",
        "tx_time_seconds",
        "tx_time_days",
        "tx_fraud",
        "tx_fraud_scenario",
        "source_file",
        F.col("source_date").alias("source_file_date"),
        "source_row_number",
        F.col("kafka_timestamp").alias("ingested_at_utc"),
        "schema_version",
        "event_id",
        F.col("produced_at").alias("produced_at_utc"),
        "kafka_key",
        "kafka_topic",
        "kafka_partition",
        "kafka_offset",
        "kafka_timestamp",
    )


def project_streaming_quarantine(
    quarantined_events: DataFrame,
) -> DataFrame:
    """Preserve rejected events, raw values and rejection reasons."""
    transaction_columns = [
        F.col(raw_column).alias(source_column)
        for source_column, raw_column in RAW_TRANSACTION_COLUMNS.items()
    ]

    return quarantined_events.select(
        *transaction_columns,
        "source_file",
        "source_file_date",
        "source_row_number",
        F.col("kafka_timestamp").alias("ingested_at_utc"),
        "schema_version",
        "event_id",
        "produced_at_utc",
        "kafka_key",
        "raw_event",
        "kafka_topic",
        "kafka_partition",
        "kafka_offset",
        "kafka_timestamp",
        "rejection_reasons",
    )
