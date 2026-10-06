# Streaming Pipeline

## Purpose

The streaming pipeline extends the batch lakehouse with local event streaming and distributed processing.

Trusted daily transaction files are replayed as individual Kafka events. Spark Structured Streaming consumes those events, preserves the original messages in Bronze, validates and deduplicates canonical transactions in Silver, quarantines rejected events, and creates event-time customer and terminal risk aggregates in Gold.

The implementation remains local-first. It runs through Docker Compose without requiring a managed Kafka service, Spark cluster, cloud account or external storage.

## Runtime Architecture

The streaming stack is defined in `compose.streaming.yaml`.

```text
Daily Pickle files
        |
        v
Transaction replay producer
        |
        v
Kafka topic: transactions.raw.v1
        |
        v
Spark ingestion job
        |
        +--> Streaming Bronze
        |
        +--> Streaming quarantine
        |
        v
Streaming Silver
        |
        v
Spark Gold job
        |
        +--> Customer risk windows
        |
        +--> Terminal risk windows
```

The ingestion and Gold transformations run as separate Structured Streaming jobs.

The ingestion job contains event-time deduplication, which is a stateful operation. The Gold job performs event-time window aggregation, which is also stateful. Persisted Silver Parquet files form the boundary between the jobs so that each stateful stage has an independent checkpoint, lifecycle and recovery process.

## Services

| Service | Responsibility |
| --- | --- |
| Kafka data init | Prepare the persistent Kafka data volume for non-root execution |
| Kafka | Run the single-node KRaft broker and expose internal and host listeners |
| Kafka init | Create the transaction topic with its required configuration |
| Transaction producer | Replay trusted daily transaction files as Kafka events |
| Spark ingestion | Parse, validate, quarantine and deduplicate transaction events |
| Spark Gold | Read persisted Silver transactions and build risk-window aggregates |

Kafka runs with one combined broker and controller because the project targets a single local development machine. The topic uses three partitions and a replication factor of one.

## Transaction Topic

The raw transaction topic is:

```text
transactions.raw.v1
```

It is created with:

- Three partitions.
- A replication factor of one.
- Seven-day retention.
- Automatic topic creation disabled at the broker.

Containers connect through the internal listener:

```text
kafka:19092
```

Host applications connect through:

```text
localhost:9092
```

The local broker uses plaintext communication and is intended only for development.

## Event Contract

Each source row is serialized as a versioned JSON event.

The event envelope contains:

| Field | Purpose |
| --- | --- |
| `schema_version` | Identify the event contract version |
| `event_id` | Provide a deterministic event identifier |
| `produced_at_utc` | Record when the replay producer created the event |
| `source_file` | Preserve the original daily filename |
| `source_file_date` | Preserve the date represented by the source file |
| `source_row_number` | Preserve the zero-based row position |
| `transaction` | Contain the original transaction fields |

The current schema version is:

```text
1.0
```

The Kafka record key is the transaction identifier. This gives transactions with the same identifier a stable partitioning key.

Event identifiers are deterministic for the source file, source date and row number. Replaying the same source data therefore reproduces the same event identities.

## Spark Runtime

`Dockerfile.spark` provides a non-root runtime containing:

- Spark 4.2.0.
- Python 3.11.
- Java 17.
- The installed `fraud-lakehouse` package.
- The operating-system libraries required by the project.

The Kafka Structured Streaming connector is:

```text
org.apache.spark:spark-sql-kafka-0-10_2.13:4.2.0
```

Connector artifacts are cached in the named `spark-ivy` volume so they do not need to be downloaded on every container start.

Both Spark jobs run locally with two worker threads. Shuffle partitions are limited to two to keep resource use appropriate for a development machine.

## Streaming Ingestion

The ingestion job performs the following stages:

1. Read Kafka records and preserve Kafka metadata.
2. Parse each JSON value with an explicit Spark schema.
3. Project the event envelope and original transaction fields.
4. Convert transaction fields to canonical data types.
5. Accumulate validation failures as rejection reasons.
6. Route invalid events to quarantine.
7. Apply an event-time watermark to valid transactions.
8. Deduplicate valid transactions by `transaction_id`.
9. Persist Bronze, Silver and quarantine outputs independently.

The default transaction watermark delay is:

```text
1 day
```

Deduplication uses `tx_datetime` as event time and retains bounded state through `dropDuplicatesWithinWatermark`.

Bronze remains immutable and preserves every received Kafka record, including duplicate and malformed events. Deduplication is applied only to the validated Silver path.

## Data Quality and Quarantine

Events are rejected when their envelope or transaction data violates the streaming contract.

Validation covers:

- Valid JSON parsing.
- Supported schema version.
- Required event identifiers and source metadata.
- UTC production timestamps.
- Required transaction fields.
- Canonical integer, timestamp and amount conversions.
- Valid fraud-label and scenario ranges.
- Consistency between the Kafka key and transaction identifier.

Rejected events retain:

- Original transaction values.
- Raw event JSON.
- Kafka topic, partition and offset.
- Kafka timestamp.
- Source metadata.
- All applicable rejection reasons.

This allows invalid events to be investigated without interrupting the valid transaction stream.

## Streaming Silver

Silver contains valid, typed and deduplicated transactions together with their lineage metadata.

Fraud labels remain in Silver because the synthetic dataset provides them and they are useful for offline evaluation. They are not used by the operational Gold risk aggregates.

Each Silver row retains:

- Canonical transaction fields.
- Source file and row metadata.
- Event schema version and identifier.
- Production and ingestion timestamps.
- Kafka key, topic, partition and offset.

## Streaming Gold

The Gold job reads persisted Silver Parquet files as a streaming source.

It creates separate customer and terminal risk aggregates using one-hour event-time windows and a ten-minute watermark.

Customer risk windows include:

- Transaction count.
- Total transaction amount.
- Average transaction amount.
- Maximum transaction amount.
- Approximate distinct terminal count.
- Night-time transaction count.

Terminal risk windows include:

- Transaction count.
- Total transaction amount.
- Average transaction amount.
- Maximum transaction amount.
- Approximate distinct customer count.
- Night-time transaction count.

Fraud labels are deliberately excluded from these operational aggregates to prevent target leakage.

Gold uses append output mode. A completed window is published only after the watermark has advanced beyond the window boundary.

## Persistent Outputs

The local `data/` directory is mounted into both Spark services.

| Dataset | Host path |
| --- | --- |
| Streaming Bronze | `data/streaming/lakehouse/bronze` |
| Streaming Silver | `data/streaming/lakehouse/silver` |
| Streaming quarantine | `data/streaming/lakehouse/quarantine` |
| Customer risk windows | `data/streaming/lakehouse/gold/customer_risk_windows` |
| Terminal risk windows | `data/streaming/lakehouse/gold/terminal_risk_windows` |
| Ingestion checkpoints | `data/streaming/checkpoints/ingestion` |
| Gold checkpoints | `data/streaming/checkpoints/gold` |

Generated streaming outputs, metadata and checkpoints are excluded from Git.

## Build the Streaming Runtime

Build the Spark services:

```bash
docker compose \
  --file compose.streaming.yaml \
  --profile pipeline \
  build \
  spark-ingestion \
  spark-gold
```

The transaction producer uses the existing application image and can be built with:

```bash
docker compose \
  --file compose.streaming.yaml \
  --profile replay \
  build \
  transaction-producer
```

## Start the Streaming Pipeline

Start Kafka and both Spark jobs:

```bash
docker compose \
  --file compose.streaming.yaml \
  --profile pipeline \
  up \
  --detach \
  --wait \
  kafka \
  spark-ingestion \
  spark-gold
```

The dependency chain initializes the Kafka data directory and creates the transaction topic before Spark ingestion starts.

Check the service states:

```bash
docker compose \
  --file compose.streaming.yaml \
  --profile pipeline \
  ps \
  --all
```

Follow the Spark logs:

```bash
docker compose \
  --file compose.streaming.yaml \
  --profile pipeline \
  logs \
  --follow \
  spark-ingestion \
  spark-gold
```

## Replay Transactions

Place trusted daily Pickle files in:

```text
data/raw
```

Filenames must use the existing `YYYY-MM-DD.pkl` contract.

Run the replay producer with its default rate of 100 events per second:

```bash
docker compose \
  --file compose.streaming.yaml \
  --profile replay \
  run \
  --rm \
  transaction-producer
```

The default command remains in the foreground until every discovered source row has been sent. A complete dataset replay at 100 events per second can take several hours. Use `--max-events` for a bounded validation run or `--events-per-second 0` to remove rate limiting.

The replay rate and event limit can be overridden:

```bash
docker compose \
  --file compose.streaming.yaml \
  --profile replay \
  run \
  --rm \
  transaction-producer \
  replay-transactions \
  --raw-dir /app/data/raw \
  --bootstrap-servers kafka:19092 \
  --topic transactions.raw.v1 \
  --events-per-second 25 \
  --max-events 1000
```

Use an event rate of zero to disable rate limiting:

```text
--events-per-second 0
```

## Inspect the Kafka Topic

Describe the transaction topic:

```bash
docker compose \
  --file compose.streaming.yaml \
  exec \
  kafka \
  /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server kafka:19092 \
  --describe \
  --topic transactions.raw.v1
```

Consume raw events for debugging:

```bash
docker compose \
  --file compose.streaming.yaml \
  exec \
  --no-TTY \
  kafka \
  /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server kafka:19092 \
  --topic transactions.raw.v1 \
  --from-beginning \
  --max-messages 5 \
  --timeout-ms 10000 \
  --formatter-property print.key=true \
  --formatter-property key.separator=" | "
```

## Checkpoint Recovery

Each output query uses its own checkpoint directory.

The checkpoints preserve:

- Kafka source offsets.
- File-source progress.
- Deduplication state.
- Event-time aggregation state.
- Sink commit metadata.

Stopping and recreating the Spark containers does not replay completed offsets or duplicate committed Silver and Gold records.

The default `docker compose down` command removes containers and the Compose network but preserves Kafka data and the Spark Ivy cache.

## Stop the Streaming Stack

Stop and remove the streaming containers:

```bash
docker compose \
  --file compose.streaming.yaml \
  --profile pipeline \
  --profile replay \
  down
```

Persistent named volumes and generated lakehouse data remain available for the next run.

## Local Reset

A reset discards streaming outputs and checkpoint history. Stop the stack before resetting:

```bash
docker compose \
  --file compose.streaming.yaml \
  --profile pipeline \
  --profile replay \
  down
```

Remove the generated streaming directory only when a fresh local run is intentionally required:

```bash
rm -r -- data/streaming
```

Removing checkpoints while retaining existing outputs can cause duplicate or inconsistent results. Outputs and their corresponding checkpoints must be reset together.

Kafka data is stored separately in the `fraud-lakehouse-streaming_kafka-data` named volume and is not removed by the standard shutdown command.

## Automated Validation

Continuous integration performs the following streaming checks:

- Validate the resolved streaming Compose configuration.
- Build the custom Spark image.
- Verify the Spark runtime.
- Verify non-root container execution.
- Validate both streaming job entry points.
- Parse explicit transaction-event schemas.
- Validate canonical type conversion.
- Validate accumulated data-quality failures.
- Validate event-time deduplication.
- Validate Bronze, Silver and quarantine projections.
- Validate restart-safe Parquet persistence.
- Validate customer and terminal risk aggregates.
- Validate the persisted Silver-to-Gold streaming boundary.

The Python unit and integration suite continues to run independently in the quality job.

## Local-Development Limitations

The streaming environment intentionally prioritizes reproducibility and learning over production-scale availability.

It currently uses:

- One Kafka broker and controller.
- A replication factor of one.
- Plaintext Kafka listeners.
- Local Spark execution.
- Local Parquet storage.
- Local Docker volumes and bind mounts.
- No schema registry.
- No distributed Spark cluster.
- No online model-serving endpoint.

These constraints are appropriate for a local portfolio implementation but would require different infrastructure, security and availability controls in production.