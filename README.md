# LegoNet

**Modular deep-learning tools for plant phenotyping from field imagery**

LegoNet detects agricultural objects and estimates biologically meaningful
properties directly from images—such as berry counts for countable grape
clusters and grapevine-root length, diameter, and color.

The project combines object detection, keypoint estimation, regression, and
multi-branch attribute prediction in a reproducible PyTorch pipeline. It
includes automated dataset and checkpoint downloads, CPU-tested command-line
tools, training, inference, evaluation, visualization, and a Streamlit
interface.

The underlying methods have been published in
[*Computers and Electronics in Agriculture*](https://doi.org/10.1016/j.compag.2024.109457),
[*Plant Phenomics*](https://doi.org/10.34133/plantphenomics.0132), and
[*Remote Sensing*](https://doi.org/10.3390/rs13132496).

## Highlights

- End-to-end training, inference, and evaluation in PyTorch.
- Direct per-image estimation, or object detection followed by per-object
  counting or attribute estimation.
- Regression- and keypoint-based architectures.
- Demonstrated on two public datasets: berry counting for countable grape
  clusters, and multi-task estimation of root length, diameter, and color.
- Automated, checksum-verified dataset and checkpoint downloads.
- Validated command-line tools and a local Streamlit interface.
- Modular checkpoint splitting, composition, and inspection utilities.
- CPU test coverage and continuous integration through GitHub Actions.

## Datasets

- `roots` uses the [Dataset of Grapevine roots with length, diameter, and color
  annotations](https://doi.org/10.5281/zenodo.8084106).
- `grapes` uses the [Embrapa Wine Grape Instance Segmentation Dataset
  (Embrapa WGISD)](https://github.com/thsant/wgisd).

See the dataset-specific documentation for complete runtime layouts and
annotation details:

- [Grape annotations and images](src/legonet/resources/datasets/Embrapa%20WGISD/README.md)
- [Grapevine-root images and annotations](src/legonet/resources/datasets/Grapevines%20data/README.md)

When publishing results, cite both the relevant LegoNet method paper and the
source dataset. See [Licensing and citation](#licensing-and-citation) below.

## Visual examples

### Per-object estimation - Grape berry counting

![Raw grapes image, detected cluster crop, ground-truth keypoint heatmap, and LegoNet-predicted keypoint heatmap](docs/images/per-object-counting-keypoint-example.jpg)

*Per-object grape berry counting. The full image (`CFR_1667.jpg` from the
[Embrapa WGISD dataset](https://github.com/thsant/wgisd)) shows (a) countable
grape-cluster annotations in blue and predicted bounding boxes in red; (b) a
detected cluster, which is cropped and passed to the keypoint-based berry-counting
estimator; and (c) and (d) the ground-truth and predicted keypoint heatmaps for
the cropped cluster, respectively.*

### Per-image estimation - Total root length

![Raw underground image, ground-truth root keypoint heatmap, and LegoNet-predicted keypoint heatmap](docs/images/trl-keypoint-heatmaps-example.jpg)

*Total root length (TRL) estimation from the underground image
`T025_L084_2012.10.10_115421_002.jpg` from the
[grapevine-root dataset](https://doi.org/10.5281/zenodo.8084106): (a) raw input;
(b) ground-truth keypoint heatmap (TRL: 124.09 mm); and
(c) predicted keypoint heatmap (TRL: 110.03 mm).*

The experiment scope, distinction between direct per-image estimation and
per-root aggregation, and staged verification plan are documented in
[`docs/reproduction.md`](docs/reproduction.md).

## Quick start

In a fresh Python 3.12 environment, clone the repository and install the CPU
build of PyTorch followed by LegoNet:

```bash
git clone https://github.com/Faina-Kh/LegoNet.git Code
cd Code
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
```

Run keypoint-based grape counting on the public test set:

```bash
legonet \
  --dataset-name grapes \
  --network-type per_object_counting \
  --estimate-type keypoints \
  --run-script Inference \
  --val-set Test \
  --have-gt true
```

On the first run, LegoNet downloads the required public dataset files and
matching pretrained checkpoint, verifies them, and caches them in the storage
directory. See the detailed environment and storage sections below for CUDA,
custom paths, and development setup.

## Streamlit GUI

For an interactive alternative to the command line, start the local GUI:

```bash
streamlit run apps/streamlit_app.py
```

The GUI exposes dataset, network, estimator, checkpoint, evaluation, and
visualization settings while previewing the exact CLI command before each run.
For a typical inference run:

1. Keep the source-checkout storage directory or select another location.
2. Choose `grapes` or `roots`, then choose the network and estimate type.
3. Leave **Weights loading** set to **Automatic pretrained weights
   (recommended)**.
4. Choose `Test` or `Val`, enable visualizations if required, inspect the
   command preview, and select **Run LegoNet**.

The storage path can be entered manually or selected with the native folder
picker when Streamlit is running locally. Checkpoints can likewise be selected
by server path or uploaded through the browser. Uploaded files are copied to a
temporary server-side directory before LegoNet starts, so uploads also work
with a remote Streamlit server.

Leave automatic dataset downloads enabled for the first public-dataset run.
The live output reports dataset and checkpoint download progress, checksum
verification, and the experiment output location. Later runs reuse the cached
files.

## Supported configurations

`--network-type` selects the task and where estimation is performed:

- `bbox_detection` detects objects but does not estimate their attributes.
- `per_object_counting` detects objects and counts annotated parts within each
  detected object.
- `per_image_estimation` predicts an attribute directly from the complete
  image without first detecting individual objects. The released direct
  per-image examples and checkpoints estimate total root length (TRL).
- `per_object_attributes` detects roots and estimates length, diameter, and
  color for each detected root using the selected estimator architecture.
- `per_object_attributes_multibranch` is the keypoint-only per-root variant
  with an additional feature and keypoint-detection branch.

The available combinations are:

| Dataset | Network type | Estimate type |
|---|---|---|
| `grapes` | `bbox_detection` | Not used by the detector |
| `grapes` | `per_object_counting` | `keypoints` or `regression` |
| `roots` | `bbox_detection` | Not used by the detector |
| `roots` | `per_image_estimation` | `keypoints` or `regression` |
| `roots` | `per_object_attributes` | `keypoints` or `regression` |
| `roots` | `per_object_attributes_multibranch` | `keypoints` |

The `--estimate-type` option selects one of two estimator architectures:

- `keypoints` uses explicit keypoint detection. The model predicts keypoint
  heatmaps and derives the requested count or attribute from those intermediate
  predictions.
- `regression` predicts the count or attribute directly from feature-pyramid
  representations, without producing keypoint heatmaps.

Training requires ground-truth annotations and uses the `Val` validation split.
Unsupported combinations fail before model or dataset construction.

## Environment setup

LegoNet provides separate setup paths for lightweight CPU development and
CUDA research runs. The CPU environment is sufficient for package validation,
the public CLI, and the default automated test suite. Full training and
research inference use the CUDA environment below.

For the recommended four-folder workspace, clone the repository as `Code`:

```bash
mkdir LegoNet
cd LegoNet
git clone https://github.com/Faina-Kh/LegoNet.git Code
cd Code
```

LegoNet then keeps source files and runtime files separate:

```text
LegoNet/
├── Code/          Git repository and installed project source
├── Datasets/      Downloaded datasets and copied annotation metadata
├── ExpResults/    Training and inference results
└── checkpoints/   Downloaded pretrained weights
```

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

When the source checkout directory is named `Code`, LegoNet uses its parent as
the default storage directory. This produces the four-folder `LegoNet/`
workspace shown above. The small grape annotation files, licenses, and dataset
notes are packaged with the code (`src/legonet/resources/datasets`) and copied
into `Datasets/` during first-time setup.
You can therefore run the quick-start inference command above from `Code/` without
`--storage-path`.

For compatibility, a checkout with another directory name continues to use
its repository root as storage. Clone or rename it as `Code`, or pass
`--storage-path` explicitly, to use the sibling-folder layout.

To keep runtime files elsewhere, supply a storage root on each command:

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

LegoNet uses a fixed directory structure beneath the selected storage root
because dataset, result, and checkpoint paths are constructed from these
directory names. Missing directories are created automatically.

The default layout for a checkout named `Code` is:

```text
LegoNet/
├── Code/
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
├── ExpResults/
    ├── grapes/
    └── roots/
└── checkpoints/
    └── zenodo-21966953/
```


## Dataset downloads

If the selected public dataset is incomplete,
LegoNet announces the source and destination before downloading it. For grapes,
only the 300 resized `.jpg` files referenced by the tracked LegoNet split
annotations are downloaded from WGISD; upstream `.txt` and `.npz` files are
skipped. For roots, the published ZIP is downloaded from Zenodo, checksum
verified, safely extracted, and checked for the expected split files.
Pass `--download-missing-data false` to disable network-based dataset setup.

Datasets can also be prepared or checked without starting inference:

```bash
legonet-data download grapes
legonet-data download roots
legonet-data download all
legonet-data verify all
```

From a source checkout, use
`python scripts/download_datasets.py download grapes`; substitute `roots` or
`all` as needed.

The source-checkout equivalent of `legonet` is
`python scripts/run_legonet.py`.

## Training from the command line

This example trains the grape counting regression model while initializing its
frozen detector from an explicitly supplied checkpoint:

```bash
python scripts/run_legonet.py \
  --storage-path /path/to/legonet-storage \
  --dataset-name grapes \
  --network-type per_object_counting \
  --estimate-type regression \
  --run-script Training \
  --val-set Val \
  --have-gt true \
  --weights-mode detector_only \
  --bbox-weights-file /path/to/legonet_bbox_grapes.pt \
  --num-of-epochs 300
```

Boolean CLI values accept `true`, `false`, `yes`, `no`, `1`, or `0`.

## Evaluation visualizations

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

## Pretrained weights

All checkpoints provided for the supported grape and grapevine-root example
configurations are available from the
[LegoNet Zenodo record](https://doi.org/10.5281/zenodo.21966953). With
`--weights-mode auto`, the inference CLI and Streamlit GUI select the matching
full-model checkpoint, verify its checksum, and cache it locally.

Use another loading mode to override automatic selection:

| Mode | Required path options | Intended use |
|---|---|---|
| `auto` | None | Download or reuse the matching published full-model checkpoint. This is the inference default. |
| `full` | `--full-weights-file` | Resume or run a complete model checkpoint. |
| `partial` | `--bbox-weights-file` and, for per-object networks, `--per-object-weights-file` | Assemble a model from task-specific checkpoints. |
| `detector_only` | `--bbox-weights-file` | Train a new per-object head using a pretrained frozen detector. Available only for per-object training. |
| `none` | None | Initialize the selected model without loading a checkpoint. |

For the checkpoint inventory, experiment limitations, checksums, and utilities
for splitting, combining, inspecting, or cleaning checkpoints, see
[`docs/pretrained_weights.md`](docs/pretrained_weights.md).

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
        resources/          Packaged annotations, licenses, and dataset notes
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
CITATION.cff                Software and preferred-paper citation metadata
```

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

## Licensing and citation

If you use LegoNet, cite the relevant method paper and every source dataset
used in your work. Machine-readable software and preferred-paper citation
metadata is available in [`CITATION.cff`](CITATION.cff). Dataset records,
licenses, and associated publications are listed below.

### Roots - Dataset of Grapevine Roots

The active `roots` configurations use the [Dataset of Grapevine roots with
length, diameter, and color annotations](https://doi.org/10.5281/zenodo.8084106). The
dataset is distributed under the
[CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/) and is the
dataset associated with the *Computers and Electronics in
Agriculture* citation above.

See the [roots dataset README](src/legonet/resources/datasets/Grapevines%20data/README.md)
for the expected runtime layout, attribute encoding, and object-level and
image-level evaluation rules.

When using the root data, cite the
[Zenodo dataset record](https://doi.org/10.5281/zenodo.8084106) and the
associated work:
> F. Khoroshevsky, K. Zhou, A. Bar-Hillel, O. Hadar, S. Rachmilevitch,
> J. E. Ephrath, N. Lazarovitch, and Y. Edan, "A CNN-based framework for
> estimation of root length, diameter, and color from in situ minirhizotron
> images," *Computers and Electronics in Agriculture*, vol. 227, article
> 109457, 2024. <https://doi.org/10.1016/j.compag.2024.109457>

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

### Grapes - Embrapa WGISD

The active `grapes` configurations use the [Embrapa Wine Grape Instance
Segmentation Dataset (WGISD)](https://github.com/thsant/wgisd). WGISD is distributed under the
[CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/).

Release copies of these inputs, their class mapping, and dataset-specific
documentation are included in
[`src/legonet/resources/datasets/Embrapa WGISD/`](src/legonet/resources/datasets/Embrapa%20WGISD/).

The combined files retain all 300 WGISD images across the three splits, while
berry-point rows are available for a subset of 111 images. These files are not
generated by LegoNet, and downloading the original WGISD annotation files does
not reproduce them. At runtime, the active grape configuration removes images
without contributed berry points and removes bounding boxes containing no
contributed berry points.

When using the grape data, cite the upstream WGISD dataset according to its
[citation instructions and permanent dataset record](https://doi.org/10.5281/zenodo.3361736),
as well as the associated LegoNet grape-counting work:

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
