# Contributing

Thanks for your interest in improving climate-analyzer!

## Development setup

```bash
git clone https://github.com/a169n/applied_software_3.git
cd applied_software_3
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Before opening a pull request

Run the same checks as the CI pipeline:

```bash
ruff check .            # lint
ruff format .           # auto-format
mypy                    # static types
pytest --cov            # tests, coverage must stay >= 90 %
```

## Workflow

1. Open an issue describing the bug or feature (templates are provided).
2. Create a branch from `main`: `git switch -c feature/short-description`.
3. Make small, focused commits with imperative messages
   (`Add Sen's slope to trend report`).
4. Open a pull request; CI must be green before merging.
5. Releases are made by tagging `main` with `vX.Y.Z` matching the version in
   `pyproject.toml` and `src/climate_analyzer/__init__.py`.

## Scientific changes

New statistical methods should cite a reference (paper, WMO guideline or
textbook) in the docstring and come with a test that checks the method on
synthetic data with a known answer.
