from datetime import UTC, datetime
from pathlib import Path

import pytest

from fraud_lakehouse import pipeline


def test_run_batch_pipeline_connects_all_layers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    timestamp = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)

    bronze_file = output_dir / "bronze" / "2018-04-01.parquet"
    silver_file = output_dir / "silver" / "2018-04-01.parquet"
    quarantine_file = output_dir / "quarantine" / "2018-04-01.parquet"
    gold_file = output_dir / "gold" / "transaction_features.parquet"

    def fake_bronze(
        actual_raw_dir: Path,
        actual_bronze_dir: Path,
        actual_timestamp: datetime,
    ) -> list[Path]:
        assert actual_raw_dir == raw_dir
        assert actual_bronze_dir == output_dir / "bronze"
        assert actual_timestamp == timestamp
        return [bronze_file]

    def fake_silver(
        actual_bronze_dir: Path,
        actual_silver_dir: Path,
        actual_quarantine_dir: Path,
    ) -> list[tuple[Path, Path]]:
        assert actual_bronze_dir == output_dir / "bronze"
        assert actual_silver_dir == output_dir / "silver"
        assert actual_quarantine_dir == output_dir / "quarantine"
        return [(silver_file, quarantine_file)]

    def fake_gold(
        actual_silver_dir: Path,
        actual_gold_dir: Path,
    ) -> dict[str, Path]:
        assert actual_silver_dir == output_dir / "silver"
        assert actual_gold_dir == output_dir / "gold"
        return {"transaction_features": gold_file}

    monkeypatch.setattr(pipeline, "ingest_bronze_directory", fake_bronze)
    monkeypatch.setattr(pipeline, "ingest_silver_directory", fake_silver)
    monkeypatch.setattr(pipeline, "write_gold_tables", fake_gold)

    result = pipeline.run_batch_pipeline(
        raw_dir,
        output_dir,
        timestamp,
    )

    assert result.bronze_files == (bronze_file,)
    assert result.silver_files == (silver_file,)
    assert result.quarantine_files == (quarantine_file,)
    assert result.gold_tables == {"transaction_features": gold_file}


def test_run_batch_pipeline_rejects_naive_timestamp(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        pipeline.run_batch_pipeline(
            tmp_path / "raw",
            tmp_path / "processed",
            datetime(2026, 9, 8, 12, 0),
        )
