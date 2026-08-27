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

- Modular deep-learning architecture supporting direct per-image estimation
  and detection-based per-object counting and attribute estimation.
- End-to-end training, inference, evaluation, and visualization in PyTorch.
- Interchangeable keypoint- and regression-based estimation approaches.
- Demonstrated on two public datasets for grape berry counting and multi-task
  estimation of root length, diameter, and color.
- Validated command-line tools and a local Streamlit interface.
- CPU test coverage and continuous integration through GitHub Actions.

## Datasets

- `roots` uses the [Dataset of Grapevine roots with length, diameter, and color
  annotations](https://doi.org/10.5281/zenodo.8084106).
- `grapes` uses the [Embrapa Wine Grape Instance Segmentation Dataset
  (Embrapa WGISD)](https://github.com/thsant/wgisd).

See the dataset-specific guides for complete runtime layouts, annotation
details, evaluation scope, and reference results:

- [Embrapa WGISD grapes](docs/datasets/grapes_embrapa-wgisd.md)
- [Grapevine roots](docs/datasets/roots_grapevine.md)

## Visual examples

### Per-object counting - Grape berries

![Raw grapes image, detected cluster crop, ground-truth keypoint heatmap, and LegoNet-predicted keypoint heatmap](docs/images/per-object-counting-keypoint-example.jpg)

*Per-object grape berry counting. The full image (`CFR_1667.jpg` from the
[Embrapa WGISD dataset](https://github.com/thsant/wgisd)) shows (a) countable
grape-cluster annotations in blue and predicted bounding boxes in red; (b) a
detected cluster, which is cropped and passed to the keypoint-based
berry-counting estimator; and (c) and (d) the corresponding ground-truth and
predicted keypoint heatmaps, respectively.*

#### Architecture options

![LegoNet object-detection pipeline and keypoint- or regression-based architectures for per-object grape berry counting](docs/images/per-object-count-models.jpg)

*Architecture options for per-object grape berry counting. A shared RetinaNet
detector locates grape clusters and RoI Align extracts each predicted crop. The
crop is then processed by either the keypoint-based or regression-based count
estimator selected with `--estimate-type`.*

### Per-image estimation - Total root length

![Raw underground image, ground-truth root keypoint heatmap, and LegoNet-predicted keypoint heatmap](docs/images/trl-keypoint-heatmaps-example.jpg)

*Total root length (TRL) estimation from the underground image
`T025_L084_2012.10.10_115421_002.jpg` from the
[grapevine-root dataset](https://doi.org/10.5281/zenodo.8084106): (a) raw input;
(b) ground-truth keypoint heatmap (TRL: 124.09 mm); and
(c) predicted keypoint heatmap (TRL: 110.03 mm).*

#### Architecture options

<p align="center">
  <img src="docs/images/per-image-TRL-models.jpg" width="75%" alt="Keypoint-based D+R and regression-based MSR architectures for direct per-image total root length estimation">
</p>

*Direct per-image TRL estimation. The complete image is processed without
first detecting individual roots. The `keypoints` option uses the Find + D+R
architecture, while the `regression` option uses the MSR architecture.*

### Per-object attribute estimation - Roots

#### Per-root attributes with standalone keypoint estimators

![Detection, crop extraction, and standalone keypoint estimators for per-root length, diameter, and color](docs/images/per-object-keypoint-model-roots.jpg)

*Standalone keypoint architecture for per-root attribute estimation. LegoNet
detects roots, extracts each predicted crop with RoI Align, and estimates
length, diameter, and color through separate keypoint-based output modules (D+R).*

##### Per-root attributes with the multibranch keypoint estimator

![Detection, crop extraction, and multibranch keypoint architecture for per-root length, diameter, and color](docs/images/per-object-multi-branch-model-roots.jpg)

*Multibranch keypoint architecture for per-root attribute estimation. Length
uses a dedicated branch, while diameter and color share a second backbone and
keypoint-detection branch.* 

Per-root predictions can subsequently be aggregated
into image-level TRL, mean diameter, and white-root fraction.

The supported configurations also include a regression-based per-root
attribute estimator, which is not shown in these two per-root diagrams.

The experiment scope, distinction between direct per-image estimation and
per-object aggregation, and staged verification plan are documented in
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

Run per-root length, diameter, and color estimation with:

```bash
legonet \
  --dataset-name roots \
  --network-type per_object_attributes_multibranch \
  --estimate-type keypoints \
  --run-script Inference \
  --val-set Test \
  --have-gt true
```
On the first run, LegoNet downloads the required public dataset files and
matching pretrained checkpoint, verifies them, and caches them in the storage
directory. See the detailed environment and storage sections below for CUDA,
custom paths, and development setup.

Unsupported combinations fail before model or dataset construction.


## Streamlit GUI

Install the optional GUI dependencies, then start the local app:

```bash
python -m pip install -e ".[gui]"
streamlit run apps/streamlit_app.py
```

The Storage path field starts with `LEGONET_STORAGE_PATH` when the environment
variable is set and otherwise uses the source-checkout root. It remains
editable and includes a **Browse local machine…**
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

For a typical inference run:

1. Keep the source-checkout storage directory or select another location.
2. Choose `grapes` or `roots`, then choose the network and estimate type.
3. Leave **Weights loading** set to **Automatic pretrained weights
   (recommended)**. The GUI shows the selected filename, download size, cache
   status, and Zenodo record before the run starts.
4. Choose `Test` or `Val`, enable visualizations if required, inspect the
   command preview, and select **Run LegoNet**.

Leave **Download missing public dataset files automatically** enabled unless
the system must remain offline. LegoNet first checks the selected local
dataset and reuses it when it is complete. If files are missing, the live
output shows JPEG download progress for grapes, or ZIP download, checksum
verification, and extraction progress for roots. A manually prepared complete
dataset is therefore not downloaded again, even when this option is enabled.

The subprocess output reports when a checkpoint download begins and when its
checksum has been verified. The first run may download approximately 125–401
MiB depending on the selected model. Later runs use the cached checkpoint.
Select **Full model checkpoint** or **Partial task checkpoints** to override
automatic selection with paths or uploaded `.pt`/`.pth` files. Select **Do not
load weights** to disable both local and downloaded checkpoints.

The GUI previews the exact CLI command before launching it.


## Environment setup

The Quick start above provides the lightweight CPU installation for normal
inference. Contributors who need an editable test environment should follow
[`CONTRIBUTING.md`](CONTRIBUTING.md). Full training and research inference use
the CUDA environment below.

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
running directly from a source checkout, including from an IDE such as PyCharm.


### Storage directory

When the source checkout directory is named `Code`, LegoNet uses its parent as
the default storage directory. This produces the four-folder `LegoNet/`
workspace shown above.
You can therefore run the quick-inference commands below from `Code/` without
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

### Folders layout

For the recommended four-folder workspace, clone the repository as `Code`:

```bash
mkdir LegoNet
cd LegoNet
git clone https://github.com/Faina-Kh/LegoNet.git Code
cd Code
```

LegoNet then keeps source files and runtime files separate.
When the source checkout is named `Code`, its parent is used as the default
storage root, producing the four-folder workspace shown below.

```text
LegoNet/
├── Code/          Git repository and installed project source
├── Datasets/      Downloaded datasets and copied annotation metadata
├── ExpResults/    Training and inference results
└── checkpoints/   Downloaded pretrained weights
```

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

For checkouts with another name, the repository root is used instead. In this case,
specify the storage root with `--storage-path` or `LEGONET_STORAGE_PATH`.

Dataset, result, and checkpoint paths are constructed from these directory names.
Missing directories are created automatically.


## Supported configurations

| Dataset | Network type | Estimate type |
|---|---|---|
| `grapes` | `bbox_detection` | Not used by the detector |
| `grapes` | `per_object_counting` | `keypoints` or `regression` |
| `roots` | `bbox_detection` | Not used by the detector |
| `roots` | `per_image_estimation` | `keypoints` or `regression` |
| `roots` | `per_object_attributes` | `keypoints` or `regression` |
| `roots` | `per_object_attributes_multibranch` | `keypoints` |

The `--estimate-type` option selects one of two estimator-based architectures:

- `keypoints` uses explicit keypoint detection. The model predicts
  keypoint heatmaps and derives the requested count or attribute from those
  intermediate predictions.
- `regression` uses direct regression from feature-pyramid representations,
  without producing keypoint heatmaps.

## Training

Training requires ground-truth annotations and uses the `Val` validation
split. Select a supported dataset, network type, and estimate type from the
table above, then use `--run-script Training`.

## Automatic dataset and checkpoint downloads

When an inference checkpoint is not supplied, LegoNet announces and downloads
the matching pretrained checkpoint from the
[LegoNet Zenodo record](https://doi.org/10.5281/zenodo.21966953). Downloads are
checksum-verified and cached in
`<storage-path>/checkpoints/zenodo-21966953/`; later runs reuse the cached file.
Pass `--full-weights-file /path/to/model.pt` to use your own full checkpoint,
or `--weights-mode none` to run without pretrained weights. Explicit local
paths always take precedence over automatic downloads.

The selected public dataset is prepared in the same way. If it is incomplete,
LegoNet announces the source and destination before downloading it.
For grapes, only the 300 resized `.jpg` files referenced by the tracked LegoNet split
annotations are downloaded from WGISD; upstream `.txt` and `.npz` files are
skipped. The small grape annotation and license files are packaged with the
code (`src/legonet/resources/datasets`) and copied into `Datasets/` during
first-time setup.
For roots, the published ZIP is downloaded from Zenodo, checksum
verified, safely extracted, and checked for the expected split files. Pass
`--download-missing-data false` to disable network-based dataset setup.

Datasets can also be prepared or checked without starting inference:

```bash
legonet-data download grapes
legonet-data download roots
legonet-data download all
legonet-data verify all
```

From a source checkout, the equivalent entry point is
`python scripts/download_datasets.py download grapes`.

The source-checkout equivalent of `legonet` is
`python scripts/run_legonet.py`.


## Checkpoint loading

The runtime supports full-model and modular checkpoint loading. See
[`docs/pretrained_weights.md`](docs/pretrained_weights.md) for the published
checkpoint inventory, provenance, limitations, checksums, and conversion
utilities.

Select one loading mode:

| Mode | Required path options | Intended use |
|---|---|---|
| `auto` | None | Download or reuse the matching published full-model checkpoint. This is the inference default. |
| `none` | None | Initialize the complete selected model without weights. |
| `full` | `--full-weights-file` | Resume or run a complete model checkpoint. |
| `partial` | `--bbox-weights-file` and, for per-object networks, `--per-object-weights-file` | Assemble a model from task-specific checkpoints. |
| `detector_only` | `--bbox-weights-file` | Train a new per-object head using a pretrained frozen detector. Available only for per-object training. |


Use `--weights-mode auto` (the inference default) for the matching published
full checkpoint. `full`, `partial`, and `detector_only` accept local paths but
download any required path that was omitted. `none` disables all checkpoint
loading. For example, this training command supplies its detector explicitly:

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


## Evaluation

The experiment scope and the distinction between direct per-image estimation
and per-object aggregation are documented in
[`docs/reproduction.md`](docs/reproduction.md).

Evaluation scope, metrics, code-output comparisons with the relevant paper results,
and known limitations can be seen in the dataset-specific documentation in
[`docs/datasets`](docs/datasets).

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


## Tests

Run the default CPU test suite:

```bash
python -m pytest -m "not slow and not gpu"
```

The suite validates installation, CLI behavior, dataset handling, checkpoint
utilities, and evaluation logic without requiring research datasets or a GPU.
GPU experiments and scientific result reproduction are documented separately
in [`docs/reproduction.md`](docs/reproduction.md). Development setup, focused
test commands, and IDE debugging are documented in
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Citation

If you use LegoNet, cite the paper associated with the model or dataset used in
your work. Machine-readable software and preferred-paper citation metadata is
available in [`CITATION.cff`](CITATION.cff). The relevant publications are
listed with each dataset below.

## Data and licensing

### Grapevine roots

The active `roots` configurations use the [Dataset of Grapevine roots with
length, diameter, and color annotations](https://doi.org/10.5281/zenodo.8084106),
created by Faina Khoroshevsky, Kaining Zhou, and Naftali Lazarovitch. The
dataset is distributed under the
[CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/) and is the
dataset associated with the preferred 2024 *Computers and Electronics in
Agriculture* citation above.

See the [roots dataset guide](docs/datasets/roots_grapevine.md)
for the expected runtime layout, attribute encoding, and object-level and
image-level evaluation rules.

Cite the associated work as:
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

### Grapes

The active `grapes` configurations use the [Embrapa Wine Grape Instance
Segmentation Dataset (WGISD)](https://github.com/thsant/wgisd).
WGISD is distributed under the
[CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/).

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

Release copies of these inputs and their class mapping are included in
[`src/legonet/resources/datasets/Embrapa WGISD/`](src/legonet/resources/datasets/Embrapa%20WGISD/).
Dataset-specific documentation is available in the
[grape dataset guide](docs/datasets/grapes_embrapa-wgisd.md).

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
