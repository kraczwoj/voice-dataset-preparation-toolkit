from __future__ import annotations

import tomllib
from pathlib import Path

from .models import ProcessingOptions


def load_options(path: Path | None) -> ProcessingOptions:
    if path is None:
        options = ProcessingOptions()
    else:
        with path.open("rb") as stream:
            payload = tomllib.load(stream)
        section = payload.get("processing", payload)
        allowed = set(ProcessingOptions.__dataclass_fields__)
        unknown = set(section) - allowed
        if unknown:
            raise ValueError(f"Unknown processing option(s): {', '.join(sorted(unknown))}")
        options = ProcessingOptions(**section)
    options.validate()
    return options
