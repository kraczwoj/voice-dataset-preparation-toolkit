# Contributing

Contributions should keep the pipeline transparent, local-first, and safe for irreplaceable source recordings.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

## Expectations

- Never overwrite source audio.
- Add tests for changes to DSP or file I/O.
- Document assumptions and measurement limitations.
- Reject unsupported formats explicitly rather than guessing.
- Keep network access and telemetry out of the core package.
- Do not commit personal, customer, or biometric voice data.
- Use clear commit messages and keep pull requests focused.

DSP changes should include a description of the expected signal behavior and a deterministic synthetic test where practical.

