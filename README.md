# LegoNet

LegoNet is a research codebase for object detection and image- or object-level
estimation in plant phenotyping. It contains the training, inference,
evaluation, data-loading, and checkpoint utilities used for grape and
grapevine-root experiments.

The repository is being prepared for a public research release. The cleanup
aims to improve reproducibility and usability without changing established
model behavior.

## Status

Available now:

- Detection, per-image estimation, and per-object estimation models.
- Training and inference through a validated command-line interface.
- A local Streamlit runner.
- Characterization and smoke tests for the active execution paths.
- Utilities for full-model and modular checkpoints.

Still being prepared:

- Curated pretrained checkpoints.
- Final dataset download links and paper citations.
- Final release metadata.

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
        runner.py           Training/inference dispatch
        models/             Model variants
        eval/               Evaluation code
scripts/
    debug_legonet.py        Editable same-process IDE debug entry point
    run_legonet.py          Thin public command-line entry point
notebooks/                  Demonstrations and experiments
tests/                      Unit, characterization, and smoke tests
environment.yml             Reproducible Conda environment
pyproject.toml              Python package and CLI metadata
LICENSE                     BSD-3-Clause source-code license
```

## Environment setup

The supplied environment targets Python 3.12, PyTorch 2.5.1, and CUDA 12.1.
Create it with Conda or Mamba:

```bash
conda env create -f environment.yml
conda activate legonet
python -m pip install --no-deps -e .
```

The CUDA-specific PyTorch wheels in `environment.yml` require a compatible
NVIDIA driver. CPU-only and other CUDA configurations require an appropriate
PyTorch installation for that machine.

Check the public CLI without starting an experiment:

```bash
legonet --help
```

The thin `python scripts/run_legonet.py` entry point remains available for
running directly from a source checkout.

## Storage directory

LegoNet requires an explicit storage root. Supply it on each command:

```bash
python scripts/run_legonet.py --storage-path /path/to/legonet-storage --help
```

or set `LEGONET_STORAGE_PATH` before running the CLI or Streamlit app.

PowerShell, current terminal:

```powershell
$env:LEGONET_STORAGE_PATH = "D:\LegoNet"
```

PowerShell, persistent for the current Windows user:

```powershell
[Environment]::SetEnvironmentVariable(
    "LEGONET_STORAGE_PATH",
    "D:\LegoNet",
    "User"
)
```

Open a new terminal after setting a persistent variable.

The storage root must already exist. The current data loaders expect this
high-level layout:

```text
legonet-storage/
├── Datasets/
│   ├── Embrapa WGISD/
│   │   ├── train.txt
│   │   ├── val.txt
│   │   ├── test.txt
│   │   └── classes.txt
│   └── Grapevines data/
│       ├── sub_Train/
│       │   ├── Train.csv
│       │   ├── Train_pointsOutput.csv
│       │   └── Train_Dia_Length_Color.txt
│       ├── sub_Val/
│       └── sub_Test/
└── ExpResults/
    ├── grapes/
    └── roots/
```

Annotation files may reference additional image locations within their dataset
directory. The final data release documentation will describe the complete
per-dataset layout.

## Supported configurations

| Dataset | Network type | Estimate type |
|---|---|---|
| `grapes` | `bbox_detection` | Not used by the detector |
| `grapes` | `per_object_counting` | `withKeyPoints` or `reg_fpn_p3_p7_min_sig` |
| `roots` | `bbox_detection` | Not used by the detector |
| `roots` | `per_image_estimation_keypoints` | `withKeyPoints` |
| `roots` | `per_image_estimation_regression` | `reg_fpn_p3_p7_min_sig` |
| `roots` | `per_object_attributes` | `withKeyPoints` or `reg_fpn_p3_p7_min_sig` |
| `roots` | `per_object_attributes_multibranch` | `withKeyPoints` |

Training requires ground-truth annotations and uses the `Val` validation split.
Unsupported combinations fail before model or dataset construction.

## Command-line use

Example roots inference configuration:

```bash
python scripts/run_legonet.py \
  --storage-path /path/to/legonet-storage \
  --dataset-name roots \
  --network-type per_object_attributes \
  --estimate-type withKeyPoints \
  --run-script Inference \
  --val-set Test \
  --have-gt true \
  --load-weights true
```

Example grapes training configuration without loading an initial checkpoint:

```bash
python scripts/run_legonet.py \
  --storage-path /path/to/legonet-storage \
  --dataset-name grapes \
  --network-type per_object_counting \
  --estimate-type withKeyPoints \
  --run-script Training \
  --val-set Val \
  --have-gt true \
  --load-weights false \
  --num-of-epochs 300
```

Boolean values accept `true`, `false`, `yes`, `no`, `1`, or `0`. Explicit
false values are preserved.

Checkpoint loading and legacy checkpoint export are mutually exclusive:

```text
--load-weights true
--save-from-model-file true
```

cannot be used together.

## PyCharm debugging

For same-process debugging with normal PyCharm breakpoints, edit the
`DEBUG_SETTINGS` mapping in `scripts/debug_legonet.py`, then run or debug that
file with the `newTorchEnv` interpreter. The settings correspond to the public
CLI options and the notebook settings. Set `LEGONET_STORAGE_PATH`, or replace
the empty `storage_path` value locally, before starting an experiment.

The debug entry point calls `legonet.cli.main()` directly rather than starting
a subprocess, so breakpoints inside the package remain active. Machine-specific
paths should not be committed.

## Streamlit GUI

Start the local GUI with:

```bash
streamlit run apps/streamlit_app.py
```

The Storage path field starts with `LEGONET_STORAGE_PATH` when the environment
variable is set. It remains editable and includes a **Browse local machine…**
button. The native folder picker is a local convenience: it opens on the
machine running Streamlit and may be unavailable for remote or headless
deployments. Manual path entry remains available in those environments.

The GUI previews the exact CLI command before launching it and writes output
to the experiment directory under `ExpResults`.

## Tests

Run the test suite with:

```bash
python -m unittest discover -s tests
```

The public-entry-point smoke tests can be run independently:

```bash
python -m unittest tests.test_public_entrypoint
```

These smoke tests verify public help output and clean failures for missing or
unsupported configuration without requiring model construction.

## Data and licensing

The original WGISD images are not redistributed by this project. Obtain them
from the [WGISD repository](https://github.com/thsant/wgisd). WGISD is
distributed under the
[CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/).
Derived WGISD annotations must be used consistently with that license, and the
original dataset should be cited.

The LegoNet source code is distributed under the
[BSD 3-Clause License](LICENSE). Dataset files and derived annotations retain
their respective licensing terms and are not covered by the source-code
license.

The grapevine-root dataset links, associated paper citations, and derived-data
terms must be finalized before the public release.

## Pretrained weights

Curated pretrained weights are not yet part of the public release. The current
runtime supports full-model and modular checkpoint loading, but commands using
`--load-weights true` require the expected local checkpoint files under the
experiment storage hierarchy.

## Known limitations

- Some runtime configuration remains research-specific inside
  `src/legonet/cli.py`.
- Pretrained checkpoint locations have not yet been converted into a public
  download manifest.
- Complete dataset setup and citation instructions are pending.
- Not every historical no-ground-truth or checkpoint-conversion path has been
  validated across every model variant.

## Before public release

- Add final paper citations and dataset URLs.
- Publish and document curated checkpoints.
- Verify a clean environment setup on a second machine.
