# LegoNet development guide

This guide covers local development, testing, repository navigation, and IDE
debugging for users who want to inspect or modify a local checkout of LegoNet.
User-facing installation, datasets, training, and inference instructions remain
in the [main README](../README.md).

## Development environment

Create a fresh Python 3.12 virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Or activate it in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the official CPU builds of PyTorch and torchvision, followed by LegoNet
and its test dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[test]"
```

Validate the installation:

```bash
python -c "import legonet"
legonet --help
python -m pytest -m "not slow and not gpu"
```

Install the optional GUI dependencies when working on the Streamlit apps:

```bash
python -m pip install -e ".[gui,test]"
```

The CPU suite uses generated tensors, temporary files, and mocked runtime
components. It does not require CUDA, research datasets, pretrained weights,
or network access after installation. Passing it verifies the software
baseline; it does not reproduce paper results or validate CUDA behavior. See
[`reproduction.md`](reproduction.md) for scientific comparisons.

## Repository layout

```text
apps/
    streamlit_app.py        Local graphical runner
src/
    legonet/
        cli.py              CLI parsing and runtime configuration
        config.py           Shared runtime configuration
        manage_weights.py   Checkpoint conversion and modular weight tools
        paths.py            Dataset and result-root construction
        resources/          Packaged annotations and licenses
        runner.py           Training/inference dispatch
        models/             Model variants
        eval/               Evaluation code
scripts/
    debug_legonet.py        Editable same-process IDE debug entry point
    run_legonet.py          Thin source-checkout CLI entry point
notebooks/                  Demonstrations and experiments
tests/                      Unit, characterization, and smoke tests
environment.yml             Reproducible CUDA research environment
pyproject.toml              Package, CLI, test, and dependency metadata
```

Keep reusable implementation under `src/legonet/`, command-line entry points
thin, and notebooks limited to demonstrations or experiments. When changing
behavior locally, add or update the closest relevant test and documentation as
appropriate.

## Running tests

Run the default CPU suite with:

```bash
python -m pytest -m "not slow and not gpu"
```

Run a focused test file while developing with, for example:

```bash
python -m pytest tests/test_datasets.py
```

Tests marked `slow` or `gpu` are intentionally excluded from the default CPU
command. A local change that affects CUDA execution or scientific output should
also be checked in the appropriate research environment with its dataset,
split, checkpoint, and configuration recorded.

## PyCharm debugging

For same-process debugging with normal PyCharm breakpoints, edit the
`DEBUG_SETTINGS` mapping in `scripts/debug_legonet.py`, then run or debug that
file with the project interpreter. The settings correspond to the public CLI
options and notebook settings.

Set `LEGONET_STORAGE_PATH` in **Run → Edit Configurations → Environment
variables**. A variable set in a separate PowerShell session is not
automatically available to PyCharm. As a local alternative, set the path in
`DEBUG_SETTINGS`:

```python
"storage_path": r"D:\LegoNet",
```

Do not commit machine-specific paths or credentials.
