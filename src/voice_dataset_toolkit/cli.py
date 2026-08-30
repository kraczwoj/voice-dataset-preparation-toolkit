from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .audio_io import read_wav
from .config import load_options
from .metrics import analyze_buffer
from .models import ProcessingOptions, discover_audio_files
from .processing import process_file
from .reports import write_analysis_csv, write_analysis_json, write_manifest


def _sources(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return discover_audio_files(path, recursive)
    raise FileNotFoundError(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voiceprep",
        description="Analyze and prepare local PCM WAV voice datasets.",
    )
    parser.add_argument("--version", action="version", version="voiceprep 1.0.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Analyze WAV files without modifying them.")
    analyze.add_argument("input", type=Path)
    analyze.add_argument("--csv", type=Path, default=Path("analysis.csv"))
    analyze.add_argument("--json", type=Path, default=Path("analysis.json"))
    analyze.add_argument("--silence-threshold", type=float, default=-45.0, metavar="DBFS")
    analyze.add_argument("--no-recursive", action="store_true")

    prepare = subparsers.add_parser("prepare", help="Prepare a dataset and write a manifest.")
    prepare.add_argument("input", type=Path)
    prepare.add_argument("output", type=Path)
    prepare.add_argument("--config", type=Path)
    prepare.add_argument("--sample-rate", type=int)
    prepare.add_argument("--bit-depth", type=int, choices=(16, 24, 32))
    prepare.add_argument("--target-lufs", type=float)
    prepare.add_argument("--no-normalize", action="store_true")
    prepare.add_argument("--peak-ceiling", type=float)
    prepare.add_argument("--max-segment", type=float)
    prepare.add_argument("--min-segment", type=float)
    prepare.add_argument("--silence-threshold", type=float)
    prepare.add_argument("--trim-padding-ms", type=int)
    prepare.add_argument("--prefix", type=str)
    prepare.add_argument("--stereo", action="store_true", help="Preserve source channel count.")
    prepare.add_argument("--no-trim", action="store_true")
    prepare.add_argument("--keep-dc", action="store_true")
    prepare.add_argument("--no-recursive", action="store_true")
    prepare.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing assets with the selected prefix.",
    )
    return parser


def _override_options(options: ProcessingOptions, args: argparse.Namespace) -> ProcessingOptions:
    mapping = {
        "sample_rate_hz": args.sample_rate,
        "bit_depth": args.bit_depth,
        "target_lufs": args.target_lufs,
        "peak_ceiling_dbfs": args.peak_ceiling,
        "max_segment_seconds": args.max_segment,
        "min_segment_seconds": args.min_segment,
        "silence_threshold_dbfs": args.silence_threshold,
        "trim_padding_ms": args.trim_padding_ms,
        "prefix": args.prefix,
    }
    for name, value in mapping.items():
        if value is not None:
            setattr(options, name, value)
    if args.stereo:
        options.mono = False
    if args.no_trim:
        options.trim_silence = False
    if args.keep_dc:
        options.remove_dc = False
    if args.no_normalize:
        options.target_lufs = None
    options.validate()
    return options


def run_analyze(args: argparse.Namespace) -> int:
    files = _sources(args.input, not args.no_recursive)
    if not files:
        print("No supported PCM WAV files found.", file=sys.stderr)
        return 2
    metrics = []
    failures = 0
    for path in files:
        try:
            result = analyze_buffer(read_wav(path), path, args.silence_threshold)
            metrics.append(result)
            status = "OK" if not result.warnings else f"WARN: {', '.join(result.warnings)}"
            print(f"[{status}] {path}")
        except Exception as exc:
            failures += 1
            print(f"[ERROR] {path}: {exc}", file=sys.stderr)
    write_analysis_csv(args.csv, metrics)
    write_analysis_json(args.json, metrics)
    print(f"Analyzed {len(metrics)} file(s); {failures} failure(s).")
    print(f"Reports: {args.csv} and {args.json}")
    return 1 if failures else 0


def run_prepare(args: argparse.Namespace) -> int:
    options = _override_options(load_options(args.config), args)
    files = _sources(args.input, not args.no_recursive)
    if not files:
        print("No supported PCM WAV files found.", file=sys.stderr)
        return 2
    args.output.mkdir(parents=True, exist_ok=True)
    existing = sorted(args.output.glob(f"{options.prefix}_[0-9][0-9][0-9][0-9][0-9].wav"))
    if existing and not args.overwrite:
        raise ValueError(
            f"Output contains {len(existing)} existing asset(s) with prefix "
            f"'{options.prefix}'. Choose another directory/prefix or pass --overwrite."
        )
    assets = []
    failures = 0
    next_index = 1
    for path in files:
        try:
            produced = process_file(path, args.output, options, next_index)
            assets.extend(produced)
            next_index += len(produced)
            print(f"[OK] {path}: {len(produced)} asset(s)")
        except Exception as exc:
            failures += 1
            print(f"[ERROR] {path}: {exc}", file=sys.stderr)
    write_manifest(args.output, assets, options)
    print(f"Prepared {len(assets)} asset(s) from {len(files) - failures} source file(s).")
    print(f"Manifest: {args.output / 'manifest.json'}")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "analyze":
            return run_analyze(args)
        if args.command == "prepare":
            return run_prepare(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
