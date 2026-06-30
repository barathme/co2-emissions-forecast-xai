# Contributing

Thank you for your interest in this repository. Since this code accompanies a specific published manuscript, the primary goal of this repository is **exact reproducibility** of the reported results, not ongoing feature development. That said, contributions are welcome in the following areas:

## Reporting Issues

If you find a discrepancy between the code and the manuscript, a bug in the pipeline, or a reproducibility failure, please open an issue with:

1. The exact command you ran
2. The Python/package versions in use (`pip freeze` or `conda list`)
3. The expected vs. actual output
4. Whether you used the Docker image, conda environment, or pip install

## Extending the Analysis

If you would like to extend this work (e.g. adding new predictors, testing additional models, applying the framework to a different target variable), we encourage you to fork the repository rather than submit a pull request, since the upstream repository is intended to remain a fixed reference implementation matching the published manuscript. Feel free to cite this repository and link back to your extension.

## Pull Requests

We will accept pull requests that:

- Fix genuine bugs (with a regression test in `tests/`)
- Improve documentation or fix typos
- Improve reproducibility (e.g. pinning additional dependency versions, fixing platform-specific issues)

We will generally not accept pull requests that change the reported numerical results, since the repository is intended to reproduce a specific published analysis. If you believe there is an error affecting the reported results, please open an issue first so we can investigate and, if confirmed, document the correction in `CHANGELOG.md`.

## Development Setup

```bash
git clone <repo-url>
cd <repo-name>
pip install -r requirements.txt
pip install pytest flake8
pytest tests/
```

## Code Style

- Follow PEP 8; `flake8` is run in CI on `src/`, `scripts/`, and `tests/`.
- Document any new function with a docstring explaining its purpose and, where relevant, why a particular methodological choice was made (see existing modules for the expected level of detail).
