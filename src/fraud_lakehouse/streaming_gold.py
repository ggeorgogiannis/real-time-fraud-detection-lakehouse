from pyspark.sql import DataFrame
from pyspark.sql import functions as F

DEFAULT_GOLD_WINDOW_DURATION = "1 hour"
DEFAULT_GOLD_WATERMARK_DELAY = "10 minutes"
APPROXIMATE_DISTINCT_RSD = 0.05


def _validate_interval(
    value: str,
    name: str,
) -> None:
    if not value.strip():
        raise ValueError(f"{name} must not be empty")


def build_customer_risk_windows(
    silver_events: DataFrame,
    window_duration: str = DEFAULT_GOLD_WINDOW_DURATION,
    watermark_delay: str = DEFAULT_GOLD_WATERMARK_DELAY,
) -> DataFrame:
    """Create event-time customer risk aggregates without fraud labels."""
    _validate_interval(window_duration, "window_duration")
    _validate_interval(watermark_delay, "watermark_delay")

    aggregated = (
        silver_events.withWatermark(
            "tx_datetime",
            watermark_delay,
        )
        .groupBy(
            F.window(
                F.col("tx_datetime"),
                window_duration,
            ).alias("event_window"),
            F.col("customer_id"),
        )
        .agg(
            F.count("transaction_id").alias("transaction_count"),
            F.sum("tx_amount").alias("total_amount"),
            F.avg("tx_amount").alias("average_amount"),
            F.max("tx_amount").alias("maximum_amount"),
            F.approx_count_distinct(
                "terminal_id",
                APPROXIMATE_DISTINCT_RSD,
            ).alias("approx_unique_terminals"),
            F.sum(
                F.when(
                    F.hour("tx_datetime").between(0, 5),
                    F.lit(1),
                ).otherwise(F.lit(0))
            ).alias("night_transaction_count"),
        )
    )

    return aggregated.select(
        F.col("event_window.start").alias("window_start"),
        F.col("event_window.end").alias("window_end"),
        "customer_id",
        "transaction_count",
        "total_amount",
        "average_amount",
        "maximum_amount",
        "approx_unique_terminals",
        "night_transaction_count",
    )


def build_terminal_risk_windows(
    silver_events: DataFrame,
    window_duration: str = DEFAULT_GOLD_WINDOW_DURATION,
    watermark_delay: str = DEFAULT_GOLD_WATERMARK_DELAY,
) -> DataFrame:
    """Create event-time terminal risk aggregates without fraud labels."""
    _validate_interval(window_duration, "window_duration")
    _validate_interval(watermark_delay, "watermark_delay")

    aggregated = (
        silver_events.withWatermark(
            "tx_datetime",
            watermark_delay,
        )
        .groupBy(
            F.window(
                F.col("tx_datetime"),
                window_duration,
            ).alias("event_window"),
            F.col("terminal_id"),
        )
        .agg(
            F.count("transaction_id").alias("transaction_count"),
            F.approx_count_distinct(
                "customer_id",
                APPROXIMATE_DISTINCT_RSD,
            ).alias("approx_unique_customers"),
            F.sum("tx_amount").alias("total_amount"),
            F.avg("tx_amount").alias("average_amount"),
            F.max("tx_amount").alias("maximum_amount"),
            F.sum(
                F.when(
                    F.hour("tx_datetime").between(0, 5),
                    F.lit(1),
                ).otherwise(F.lit(0))
            ).alias("night_transaction_count"),
        )
    )

    return aggregated.select(
        F.col("event_window.start").alias("window_start"),
        F.col("event_window.end").alias("window_end"),
        "terminal_id",
        "transaction_count",
        "approx_unique_customers",
        "total_amount",
        "average_amount",
        "maximum_amount",
        "night_transaction_count",
    )
