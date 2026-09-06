# Four Crops minirhizotron root dataset

The `roots_four_crops` option uses the dataset published with *Root Length
Estimation: Automated Minirhizotron Image Analysis with Convolutional Networks
without Segmentation*. The Zenodo record contains 4,015 images divided among
four subsets and provides total root length (TRL) in millimetres together with
root-point coordinates.

## Download and installation

Zenodo record [7482146](https://doi.org/10.5281/zenodo.7482146) publishes one
file for the complete dataset:

| File | Size | MD5 |
|---|---:|---|
| `Datasets.zip` | 4,625,881,819 bytes (approximately 4.31 GiB) | `5c1be488f09c4077e5a4b15c435aa964` |

Because all four subsets are packaged in one archive, choosing a subset
reduces extracted disk usage but does not reduce the initial download. LegoNet
caches the verified archive at:

```text
<storage-path>/downloads/zenodo-7482146/Datasets.zip
```

Manual download through a browser or resumable download manager is recommended
for this dataset because the large Zenodo transfer can be slow. Download
`Datasets.zip` from the record, keep its filename unchanged, and place the
complete archive at the cache path above. Then run the normal dataset or model
command. LegoNet verifies the archive size and MD5 before extracting anything.
Do not place a partial browser download (`.part`, `.crdownload`, or equivalent)
at that path.

Automatic download is still supported and reports progress, but an interrupted
automatic transfer currently restarts from the beginning.

It then safely extracts only the requested `Dataset N` directory and
normalizes its name:

```text
<storage-path>/Datasets/Four Crops/
|-- dataset_1/
|-- dataset_2/
|-- dataset_3/
`-- dataset_4/
```

Install one subset from a source checkout with:

```cmd
python scripts\download_datasets.py download roots_four_crops ^
  --dataset-subset dataset_1
```

Later subset installations reuse the cached archive. Existing incomplete
subset directories are not overwritten or merged automatically.

If the goal is only to confirm that the direct per-image TRL keypoint or
regression model runs, use `roots_grapevines` first. Its download is much
smaller and its matching public checkpoints are selected automatically. Use
Four Crops when its crop-specific data or paper experiment is required.

## Subset roles

- Dataset 1 and Dataset 2 contain their published training, validation, and
  test directories and support training and inference.
- Dataset 3 and Dataset 4 are inference-only in LegoNet.

The published layouts are used without renaming source images or annotation
files. Dataset 1 and Dataset 2 contain:

```text
dataset_N/
|-- sub_Train/{Train_TRL.csv, Train_pointsOutput.csv, images...}
|-- sub_Val/{Val_TRL.csv, Val_pointsOutput.csv, images...}
`-- sub_Test/{Test_TRL.csv, Test_pointsOutput.csv, images...}
```

Dataset 3 stores `TRL.csv`, `pointsOutput.csv`, and its images directly in the
subset directory. Dataset 4 stores the same pair of CSV files and images in
each acquisition directory (for example, `CORN 2020_tube 16`). LegoNet builds
a small combined Dataset 4 manifest under the run's
`InputManifests/dataset_4` results directory. The generated manifest refers to
images by acquisition-relative path; the downloaded dataset is not modified.

## Annotation format

The TRL files have no header. Each row is:

```text
image_filename.jpg,total_root_length
```

The point files also have no header and use flat x/y pairs:

```text
image_filename.jpg,x1,y1,x2,y2,...
```

An image with no annotated root points is represented by a filename-only row.
Before a run, LegoNet checks that TRL is finite and nonnegative, coordinates
form integer x/y pairs, both files describe the same images, and every
referenced image exists. The published annotations include some negative
boundary coordinates; consistent with the legacy roots loader, LegoNet clamps
their affected coordinate to zero when loading the sample. These rules also
describe the annotation structure a future user-provided Four Crops-compatible
dataset will need to follow.

Select exactly one subset per run:

```cmd
python scripts\run_legonet.py ^
  --dataset-name roots_four_crops ^
  --dataset-subset dataset_3 ^
  --network-type per_image_estimation ^
  --estimate-type regression ^
  --run-script Inference ^
  --val-set Test
```

The per-image TRL loader supports both regression and keypoint estimator
types. Dataset 1 and Dataset 2 resolve their selected published split directly;
Dataset 3 and Dataset 4 accept only the `Test` split.
