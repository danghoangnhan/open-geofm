# Contributing to open-geofm

Thanks for your interest! `open-geofm` is an **educational** reproduction —
pedagogical clarity beats clever optimisation. Contributions that **improve a
learner's experience** are especially welcome.

## Quick start

```bash
git clone https://github.com/danghoangnhan/open-geofm
cd open-geofm
uv sync --extra dev
uv run pre-commit install
uv run pytest -q
```

## Workflow

1. **Open an issue** before large changes — describe what you'd improve and which
   pedagogical phase it belongs to (see [wiki/00-Overview](https://github.com/danghoangnhan/open-geofm/wiki/00-Overview)).
2. **Branch off `main`**, write your change, run the tests.
3. **Sign off your commits** (Developer Certificate of Origin):
   ```bash
   git commit -s -m "..."
   ```
   We do not require a CLA.
4. **Run the formatters before pushing**:
   ```bash
   uv run ruff check --fix src tests scripts
   uv run black src tests scripts
   ```
   The pre-commit hook does this automatically.
5. **Open a PR**. CI must be green (ruff + pytest, CPU-only).

## Style

- Python 3.12, type hints everywhere new code lands.
- `ruff` config in `pyproject.toml` (line length 100). `black` is the formatter.
- Tests use `pytest`. Mark CUDA-only tests `@pytest.mark.gpu`; CI skips them.
- Keep test files small and focused; the invariants in `tests/test_algorithm1_invariants.py` are the model.

## Documentation

- Long-form docs live in `wiki/` (synced to the GitHub Wiki by `.github/workflows/wiki-sync.yml`).
- Code docstrings should reference the relevant blueprint phase (e.g. "Blueprint §2 Phase 4").
- The Blackwell setup log (`wiki/07-Blackwell-Setup-Log.md`) is the single most useful page — please add any sm_120 landmine you hit, with the upstream issue link.

## Issue templates

- **Bug** — what you tried, what happened, what you expected.
- **Pedagogical improvement** — which page / notebook would be clearer, and how.
- **New notebook** — what concept it would teach and where it fits in the 01–10 sequence.

## Code of Conduct

By participating you agree to the [Contributor Covenant v2.1](CODE_OF_CONDUCT.md).
