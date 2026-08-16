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
data/
    Embrapa WGISD/           Processed grape annotations and data license
environment.yml             Reproducible Conda environment
pyproject.toml              Python package and CLI metadata
LICENSE                     BSD-3-Clause source-code license
CITATION.cff                Software and preferred-paper citation metadata
```

## Environment setup

LegoNet provides separate setup paths for lightweight CPU development and
CUDA research runs. The CPU environment is sufficient for package validation,
the public CLI, and the default automated test suite. Full training and
research inference use the CUDA environment below.

### CPU development and tests

Create a fresh Python 3.12 virtual environment and install the official CPU
builds of PyTorch and torchvision before installing LegoNet:

```bash
python -m venv .venv
```

On Linux or macOS, activate it with:

```bash
source .venv/bin/activate
```

On Windows PowerShell, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

Then install and validate the project:

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[test]"
python -c "import legonet"
legonet --help
python -m pytest -m "not slow and not gpu"
```

These tests use generated tensors, temporary files, and mocked runtime
components. They do not require CUDA, research datasets, pretrained weights,
or network access after installation. Passing CPU CI verifies the software
baseline; it does not reproduce the papers' experimental results or validate
CUDA performance.

### CUDA research environment

The supplied Conda environment targets Python 3.12, PyTorch 2.5.1, and CUDA
12.1. Create it with Conda or Mamba:

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
  --weights-mode full \
  --full-weights-file /path/to/per_object_attributes.pt
```

Example grapes training configuration with a pretrained detector and a newly
initialized counting head:

```bash
python scripts/run_legonet.py \
  --storage-path /path/to/legonet-storage \
  --dataset-name grapes \
  --network-type per_object_counting \
  --estimate-type reg_fpn_p3_p7_min_sig \
  --run-script Training \
  --val-set Val \
  --have-gt true \
  --weights-mode detector_only \
  --bbox-weights-file /path/to/legonet_bbox_grapes.pt \
  --num-of-epochs 300
```

For grapes per-object counting, each IoU-matched predicted crop is trained
against the full count stored for its matched GT box. Validation, checkpoint
selection, and test evaluation use the same original matched-GT count.

### Best-epoch metric

Per-object counting always minimizes count relative error when selecting the
best checkpoint. Attribute training accepts `--checkpoint-attribute` with
`length`, `diameter`, or `color`; it minimizes the corresponding relative
error for continuous attributes and the classification error rate for color.
Length is the default, preserving the published roots training behavior.
1-FVU remains a reported metric and is not used for checkpoint selection. The
Streamlit runner shows the attribute selector only for attribute networks.

### Evaluation of empty images

Bounding-box detection and per-object evaluation intentionally use different
image sets:

- Bounding-box detection evaluates every image, including images without
  ground-truth boxes. Predictions on those images are false positives.
- Per-object evaluation excludes images without usable per-object targets.
  This includes both images without ground-truth boxes and images whose
  ground-truth boxes contain no annotated points.

### Evaluation visualizations

Visualization output is controlled by the master `--to-draw` option. Two
additional options control full-image artifacts independently:

| Option | Default | Output |
|---|---:|---|
| `--draw-detection-overview` | `true` | Saves full images with GT points/boxes and predicted boxes. |
| `--draw-gt-only` | `false` | Saves annotation-debugging images containing GT points and boxes without predictions. |

For example:

```bash
python scripts/run_legonet.py \
  ... \
  --to-draw true \
  --draw-detection-overview false \
  --draw-gt-only false
```

The options affect only the full-image overview and GT-only artifacts.
Per-object predicted-box images, predicted crops, and requested keypoint maps
remain available while `--to-draw true` is selected.

The detection overview is useful for relatively sparse datasets such as
grapes. Dense roots images may contain many overlapping roots, points, GT
boxes, and predictions, making the full-image overview difficult to read. For
roots, disabling `--draw-detection-overview` while retaining per-object crop
visualizations is usually clearer.

GT-only output is intended for annotation inspection rather than routine model
evaluation. Its folder is not created unless `--draw-gt-only true` is passed.

### Weight selection

Checkpoint paths are supplied explicitly instead of inferred from a predefined
directory. Select one loading mode:

| Mode | Required path options | Intended use |
|---|---|---|
| `none` | None | Initialize the complete selected model without weights. |
| `full` | `--full-weights-file` | Resume or run a complete model checkpoint. |
| `partial` | `--bbox-weights-file` and, for per-object networks, `--per-object-weights-file` | Assemble a model from task-specific checkpoints. |
| `detector_only` | `--bbox-weights-file` | Train a new per-object head using a pretrained frozen detector. Available only for per-object training. |

The legacy `--load-weights`, `--load-only-bbox-weights`, and `--weights-type`
options remain accepted for compatibility, but new commands should use
`--weights-mode` and explicit file paths.

### Split a full checkpoint into partial weights

Use the standalone converter for per-object checkpoints saved by the current
code. It validates the selected architecture before creating separate detector
and per-object head checkpoints:

```powershell
python scripts/split_full_checkpoint.py `
  --full-weights-file "C:\weights\legonet_epoch=120.pt" `
  --network-type per_object_counting `
  --estimate-type reg_fpn_p3_p7_min_sig `
  --detector-output-file "C:\weights\legonet_bbox_grapes.pt" `
  --per-object-output-file "C:\weights\legonet_counting_reg.pt"
```

For an attributes head, pass its attribute names. Each name maps to an
`estimator_<name>` module:

```powershell
python scripts/split_full_checkpoint.py `
  --full-weights-file "C:\weights\legonet_epoch=90.pt" `
  --network-type per_object_attributes `
  --estimate-type withKeyPoints `
  --attribute-names length diameter color `
  --detector-output-file "C:\weights\legonet_bbox_roots.pt" `
  --per-object-output-file "C:\weights\legonet_attributes_kp.pt"
```

Leave `--attribute-names` out when the head uses the single module name
`estimator`. Existing files are preserved unless `--overwrite` is supplied.
The converter refuses architecture mismatches and never overwrites its source
checkpoint. `--detector-output-file` is optional; omit it when the detector
weights have already been saved and only a new per-object head file is needed.

The same operation is available in a separate Streamlit page:

```powershell
streamlit run apps/checkpoint_splitter.py
```

The converter page accepts an output directory separately from the detector
and per-object `.pt` filenames. When Streamlit runs locally, the output
directory can also be selected with the native folder browser.

To perform the reverse operation, combine compatible detector and per-object
partial checkpoints into one full checkpoint:

```powershell
python scripts/combine_partial_checkpoints.py `
  --detector-weights-file "C:\weights\legonet_bbox_grapes.pt" `
  --per-object-weights-file "C:\weights\legonet_counting_reg.pt" `
  --network-type per_object_counting `
  --estimate-type reg_fpn_p3_p7_min_sig `
  --full-output-file "C:\weights\legonet_counting_reg_full.pt"
```

The combiner validates both module sets, adds the `bbox_detection.` prefix to
detector keys, verifies the saved checkpoint, and refuses to overwrite either
source file. Named attributes use the same `--attribute-names` option as the
split operation. Combination is rejected whenever either partial checkpoint's
modules do not exactly match the modules defined by the selected network type,
estimate type, and attribute names.

The reverse operation also has a local Streamlit page:

```powershell
streamlit run apps/checkpoint_combiner.py
```

To remove one obsolete module, such as `find_2`, from a checkpoint through a
local Streamlit page, run:

```powershell
streamlit run apps/checkpoint_module_remover.py
```

The page preserves the source checkpoint, displays the modules before and
after removal, and writes the cleaned weights to a separate output file.

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

When launching the script from PyCharm, configure `LEGONET_STORAGE_PATH` in
**Run → Edit Configurations → Environment variables**. A variable set in a
separate PowerShell session is not automatically available to PyCharm. As a
local alternative, put the path directly in `DEBUG_SETTINGS`:

```python
"storage_path": r"D:\LegoNet",
```

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

Weight loading is selected by mode, and the GUI displays only the checkpoint
paths required by that mode. Each checkpoint supports manual server-path entry
or a browser-based file upload. Uploaded checkpoints are copied to a temporary
server-side directory before the LegoNet subprocess starts. This works both
locally and when the browser connects to a remote Streamlit server. Manual path
entry remains useful when the checkpoint already exists on the machine running
Streamlit.

The GUI previews the exact CLI command before launching it and writes output
to the experiment directory under `ExpResults`.

## Tests

Run the default fast CPU suite with:

```bash
python -m pytest -m "not slow and not gpu"
```

The public-entry-point smoke tests can be run independently:

```bash
python -m pytest tests/test_public_entrypoint.py
```

These smoke tests verify public help output and clean failures for missing or
unsupported configuration without requiring model construction.

Tests marked `slow` or `gpu` are intentionally excluded from the default CPU
command. GitHub Actions installs CPU-only PyTorch, verifies the installed
package and `legonet` command, collects and runs the CPU suite with coverage,
and builds the distribution artifacts. Full dataset and checkpoint
reproduction remains a separate research validation step.

## Citation

The preferred citation for the current root-length, diameter, and color models
is:

> F. Khoroshevsky, K. Zhou, A. Bar-Hillel, O. Hadar, S. Rachmilevitch,
> J. E. Ephrath, N. Lazarovitch, and Y. Edan, "A CNN-based framework for
> estimation of root length, diameter, and color from in situ minirhizotron
> images," *Computers and Electronics in Agriculture*, vol. 227, article
> 109457, 2024. <https://doi.org/10.1016/j.compag.2024.109457>

Machine-readable citation metadata is available in [`CITATION.cff`](CITATION.cff).

## Data and licensing

### Grapevine roots

The active `roots` configurations use the [Dataset of Grapevine roots with
length, diameter, and color annotations](https://doi.org/10.5281/zenodo.8084106),
created by Faina Khoroshevsky, Kaining Zhou, and Naftali Lazarovitch. The
dataset is distributed under the
[CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/) and is the
dataset associated with the preferred 2024 *Computers and Electronics in
Agriculture* citation above.

See the [roots dataset README](data/roots/README.md) for the expected runtime
layout, attribute encoding, and object-level and image-level evaluation rules.

An earlier, related collection is the [Dataset for "Root Length Estimation:
Automated Minirhizotron Image Analysis with Convolutional Networks without
Segmentation"](https://doi.org/10.5281/zenodo.7482146). It contains 4,015
minirhizotron images from four crop species and is also distributed under
CC BY 4.0. Its associated paper is:

> F. Khoroshevsky, K. Zhou, S. Chemweno, Y. Edan, A. Bar-Hillel, O. Hadar,
> B. Rewald, P. Baykalov, J. E. Ephrath, and N. Lazarovitch, "Automatic Root
> Length Estimation from Images Acquired In Situ without Segmentation,"
> *Plant Phenomics*, vol. 6, article 0132, 2024.
> <https://doi.org/10.34133/plantphenomics.0132>

### Grapes

The active `grapes` configurations use the [Embrapa Wine Grape Instance
Segmentation Dataset (WGISD)](https://github.com/thsant/wgisd), accessed for
the associated study on 23 June 2021. WGISD is distributed under the
[CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/).
The grape experiments use berry point annotations contributed by Faina
Khoroshevsky and Stanislav Khoroshevsky.

LegoNet uses the resized JPEG images published in WGISD's
[`data/` directory](https://github.com/thsant/wgisd/tree/master/data), not the
images in
[`original_resolution/`](https://github.com/thsant/wgisd/tree/master/original_resolution).
The `train.txt`, `val.txt`, and `test.txt` files required by LegoNet are
preprocessed, LegoNet-specific inputs. They were prepared separately by
combining two annotation types:

```text
image.jpg,grapes,x1,y1,x2,y2  # grape-cluster bounding box
image.jpg,grapes,x,y          # contributed berry-point annotation
```

Release copies of these inputs, their class mapping, and dataset-specific
documentation are included in
[`data/Embrapa WGISD/`](data/Embrapa%20WGISD/).

The combined files retain all 300 WGISD images across the three splits, while
berry-point rows are available for a subset of 111 images. These files are not
generated by LegoNet, and downloading the original WGISD annotation files does
not reproduce them. At runtime, the active grape configuration removes images
without contributed berry points and removes bounding boxes containing no
contributed berry points.

Cite the associated work as:

> F. Khoroshevsky, S. Khoroshevsky, and A. Bar-Hillel, "Parts-per-Object Count
> in Agricultural Images: Solving Phenotyping Problems via a Single Deep
> Neural Network," *Remote Sensing*, vol. 13, no. 13, article 2496, 2021.
> <https://doi.org/10.3390/rs13132496>

The wheat and banana datasets discussed in that paper are owned by the Israel
Phenomics Consortium and are not public. They are not redistributed by this
project.

### License boundaries

The LegoNet source code is distributed under the
[BSD 3-Clause License](LICENSE). Dataset files and derived annotations retain
their respective licensing terms and are not covered by the source-code
license.

## Pretrained weights

Curated pretrained weights are not yet part of the public release. The current
runtime supports full-model and modular checkpoint loading, but commands using
`--load-weights true` require the expected local checkpoint files under the
experiment storage hierarchy.

Available candidate checkpoints do not cover every reported scenario, and
some cannot reproduce paper results exactly because historical fold splits or
original training checkpoints are unavailable. See
[`docs/pretrained_weights.md`](docs/pretrained_weights.md) for the per-model
status, limitations, and planned verification process.

## Known limitations

- Some runtime configuration remains research-specific inside
  `src/legonet/cli.py`.
- Pretrained checkpoint locations have not yet been converted into a public
  download manifest.
- Exact reproduction of the roots paper's five-fold results is not currently
  possible because the historical fold definitions and complete fold-specific
  checkpoint set are unavailable.
- The grape architecture-ablation code and checkpoints are not included in the
  current release.
- Not every historical no-ground-truth or checkpoint-conversion path has been
  validated across every model variant.

## Before public release

- Publish and document curated checkpoints.
- Verify a clean environment setup on a second machine.
- Decide how partial checkpoint export should group and save shared and
  branch-specific module weights for the multibranch network.
- Generalize dataset and model configuration so the existing network
  architectures can be trained on other datasets, including support for a
  configurable number and list of attribute estimators rather than the three
  root-specific estimators for length, diameter, and color.
- Complete and validate the existing partial COCO-style dataloader support,
  and demonstrate it using the grape dataset's repository-provided JSON
  annotations.
