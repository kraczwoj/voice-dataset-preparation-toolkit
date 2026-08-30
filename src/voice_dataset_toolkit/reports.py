from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from .models import AudioMetrics, ProcessedAsset, ProcessingOptions


def write_analysis_csv(path: Path, metrics: Iterable[AudioMetrics]) -> None:
    rows = [item.to_dict() for item in metrics]
    for row in rows:
        row["warnings"] = "; ".join(row["warnings"])
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_analysis_json(path: Path, metrics: Iterable[AudioMetrics]) -> None:
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "files": [item.to_dict() for item in metrics],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _asset_csv_row(asset: ProcessedAsset) -> dict[str, object]:
    row: dict[str, object] = {
        "source": asset.source,
        "output": asset.output,
        "segment_index": asset.segment_index,
        "source_start_seconds": asset.source_start_seconds,
        "source_end_seconds": asset.source_end_seconds,
        "applied_gain_db": asset.applied_gain_db,
        "target_lufs": asset.target_lufs,
        "achieved_lufs": asset.achieved_lufs,
        "peak_limited_normalization": asset.peak_limited_normalization,
    }
    row.update({f"output_{key}": value for key, value in asset.metrics.to_dict().items()})
    row["output_warnings"] = "; ".join(asset.metrics.warnings)
    return row


def write_manifest(
    output_dir: Path,
    assets: list[ProcessedAsset],
    options: ProcessingOptions,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_payload = {
        "tool": "voice-dataset-preparation-toolkit",
        "generated_at": datetime.now(UTC).isoformat(),
        "options": asdict(options),
        "asset_count": len(assets),
        "assets": [asset.to_dict() for asset in assets],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(json_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    rows = [_asset_csv_row(asset) for asset in assets]
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
