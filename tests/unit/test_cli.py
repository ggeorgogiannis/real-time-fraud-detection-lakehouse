import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest

from fraud_lakehouse import cli
from fraud_lakehouse.pipeline import BatchPipelineResult


def test_main_runs_pipeline_and_logs_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    timestamp = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)

    def fake_run_batch_pipeline(
        raw_dir: Path,
        output_dir: Path,
        ingested_at_utc: datetime | None,
    ) -> BatchPipelineResult:
        assert raw_dir == tmp_path / "raw"
        assert output_dir == tmp_path / "processed"
        assert ingested_at_utc == timestamp

        return BatchPipelineResult(
            bronze_files=(output_dir / "bronze" / "daily.parquet",),
            silver_files=(output_dir / "silver" / "daily.parquet",),
            quarantine_files=(output_dir / "quarantine" / "daily.parquet",),
            gold_tables={
                "transaction_features": (output_dir / "gold" / "transaction_features.parquet")
            },
        )

    monkeypatch.setattr(cli, "run_batch_pipeline", fake_run_batch_pipeline)
    caplog.set_level(logging.INFO)

    exit_code = cli.main(
        [
            "run",
            "--raw-dir",
            str(raw_dir),
            "--output-dir",
            str(output_dir),
            "--ingested-at-utc",
            "2026-09-08T12:00:00Z",
        ]
    )

    assert exit_code == 0
    assert "pipeline_completed" in caplog.text
