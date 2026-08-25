# Embrapa WGISD grape dataset

LegoNet's `grapes` configurations use resized images from the **Embrapa Wine
Grape Instance Segmentation Dataset (WGISD)** together with LegoNet-specific
split files that combine grape-cluster boxes and berry-point annotations.
These annotations support:

- grape-cluster bounding-box detection; and
- berry counting within countable grape clusters.

The processed annotation files are included in this directory. The
corresponding images are not redistributed by LegoNet.

LegoNet uses the resized JPEG images published in WGISD's
[`data/` directory](https://github.com/thsant/wgisd/tree/master/data), not the
images in
[`original_resolution/`](https://github.com/thsant/wgisd/tree/master/original_resolution).

## Provenance

The annotations are derived from the [Embrapa Wine Grape Instance Segmentation
Dataset (WGISD)](https://github.com/thsant/wgisd). They combine:

- grape-cluster bounding boxes published by WGISD; and
- berry-point annotations contributed to WGISD by Faina Khoroshevsky and
  Stanislav Khoroshevsky.

The combined `train.txt`, `val.txt`, and `test.txt` files were created in a
separate historical preprocessing step. They retain all 300 WGISD image names
across the three splits and contain berry point annotations for the 111 images
with the relevant contributed annotations. Within the 111-image berry-annotated
subset, center dots were added only for berries belonging to countable grape
clusters. A cluster was considered countable when a human annotator could
visually count its berries; partially occluded or blurred clusters were
excluded.

## Files and annotation format

The expected files are:

```text
train.txt
val.txt
test.txt
classes.txt
```

The `train.txt`, `val.txt`, and `test.txt` files required by LegoNet are
preprocessed inputs specific to LegoNet. They were prepared separately by
combining two types of annotations. Each annotation row is comma-separated
and uses absolute pixel coordinates:

```text
image.jpg,grapes,x1,y1,x2,y2  # grape-cluster bounding box
image.jpg,grapes,x,y           # contributed berry-point annotation
```

LegoNet uses the four-column and six-column row lengths to distinguish points
from bounding boxes.
`classes.txt` maps the `grapes` class name to its numeric class identifier.

## Download and runtime layout

LegoNet automatically downloads only the resized JPEG images referenced by its
split files from WGISD's
[`data/` directory](https://github.com/thsant/wgisd/tree/master/data). It does
not download WGISD's `.txt` or `.npz` files. The download is pinned to commit
`6910edc5ae3aae8c20062941b1641821f0c30127` for reproducibility. Do not use the
images from WGISD's `original_resolution/` directory with these coordinates.

The current LegoNet loader expects the images and annotation files together in
the runtime storage directory:

```text
<storage-path>/
`-- Datasets/
    `-- Embrapa WGISD/
        |-- train.txt
        |-- val.txt
        |-- test.txt
        |-- classes.txt
        `-- <WGISD resized JPEG images>
```

Run `legonet-data download grapes` to prepare the images without starting an
experiment, or allow the normal CLI/GUI inference workflow to prepare them.
Pass `--storage-path PATH` when the destination is outside the source checkout.

## Split coverage and filtering

The tracked annotation files have the following coverage:

| Split | Images | Images with berry points | Raw bounding-box rows | Grape clusters with berry points | Berry-point rows |
|---|---:|---:|---:|---:|---:|
| Train | 218 | 62 | 3,214 | 90 | 3,956 |
| Validation | 24 | 24 | 367 | 31 | 1,466 |
| Test | 58 | 25 | 850 | 44 | 1,927 |
| Total | 300 | 111 | 4,431 | 165 | 7,349 |

The raw bounding-box counts include every WGISD grape-cluster box in the
annotation files. For the active grape configuration, LegoNet associates each
berry point with a bounding box, removes images without berry points, and
removes boxes containing no berry points. The point-annotated cluster counts
show the resulting subset used by the model.

## Supported configurations

LegoNet supports two grape tasks:

| Network type | Estimate type | Output |
|---|---|---|
| `bbox_detection` | Not used | Grape-cluster bounding boxes |
| `per_object_counting` | `keypoints` or `regression` | Berry count for each matched, countable cluster |

The keypoint estimator predicts a berry heatmap and derives the count from its
detected keypoints. The regression estimator predicts the berry count directly
from the detected cluster crop.

## Evaluation scope

Here, an **eligible grape-cluster box** means a box for a countable cluster: a
WGISD cluster box that contains contributed berry-point annotations and is
retained after LegoNet's filtering. It does not mean every original
grape-cluster box published by WGISD. Boxes for unannotated, occluded, blurred,
or otherwise non-countable clusters are excluded from the active grape tasks.

Bounding-box detection evaluates every image retained by the active grape
configuration and every eligible grape-cluster box. Retained images without
ground-truth boxes remain in detection evaluation, where their predictions
count as false positives.

Per-object counting uses only clusters with contributed berry points. Each
predicted box must first be matched to an unclaimed ground-truth cluster at the
configured IoU threshold. The matched predicted crop is evaluated against the
full berry count stored for that ground-truth box. Unmatched predictions remain
part of detection evaluation but do not enter counting metrics.

Counting reports absolute and relative count errors for matched clusters. The
same original matched-ground-truth count is used during validation, checkpoint
selection, and test evaluation.

## Paper comparison

The released detector and keypoint-based counting checkpoints are the original 
checkpoints used for the published experiments. Their current-code results are
therefore expected to remain relatively close to the reported paper values,
subject to software-version differences. 
The original MSR checkpoint was unavailable, therefor the released regression-based 
counting checkpoint was produced by a newer training run and is not the checkpoint used 
for the paper results.

### Bounding-box detection

The grape bounding-box detector is evaluated with a minimum confidence score
of `0.7`, an IoU threshold of `0.5`, and an NMS threshold of `0.3`. The current
code reproduces the paper's reported recall of 61.4%. The paper reports
approximately 49% precision, compared with 47.4% from the current code.

**Reproducibility note.** In the current PyTorch/Torchvision environment,
evaluation of the available detector checkpoint on the test set produced 27
matched ground-truth boxes among 57 detections, or 47.4% precision.
Two current false-positive detections have confidence scores immediately above
the `0.7` boundary. A threshold-sensitivity check at `0.713` removed those two
detections without removing any true positives, giving 27 matches among 55
detections, or 49.1% precision, while leaving recall unchanged. This indicates
that the precision difference is caused by detections close to the confidence
threshold, plausibly due to numerical or NMS differences between the historical
and current package versions. The documented evaluation threshold remains
`0.7`; the `0.713` result is reported only as a reproducibility diagnostic.

### Per-object counting

The following values compare the berry-count results reported in the paper
with current-code output from the released checkpoints. MRD is the mean
relative difference in the predicted berry count. MSR and D+R are the paper's
architecture names: MSR (Multiple Scale Regression) corresponds to the
`regression` estimate type, while D+R (Detection + Regression) corresponds to
the `keypoints` estimate type for explicit berry detection.

| Architecture | Estimate type | Source | Count error (MRD) | 1-FVU |
|---|---|---|---:|---:|
| MSR | Regression | Paper | 14.7% | 0.751 |
| MSR | Regression | Current code | 15.6% | 0.703 |
| D+R | Keypoints | Paper | 9.2% | 0.831 |
| D+R | Keypoints | Current code | 9.3% | 0.831 |

The paper values are from the
[LegoNet grape-counting study](https://doi.org/10.3390/rs13132496). The close
D+R comparison uses the original keypoint checkpoint. For the MSR results, minor
differences from the reported results are expected due to training variability.

## Visualization guidance

For grape images, the full detection overview is useful because it shows which
clusters are annotated as countable and whether predicted boxes match them.
Keypoint-based counting can additionally save the cropped cluster and its
predicted berry heatmap.

Use:

```text
--to-draw true
--draw-detection-overview true
--draw-gt-only false
```

Enable `--draw-gt-only true` only when inspecting annotations without model
predictions.

## Example

Run keypoint-based grape counting on the public test split with automatic
dataset and checkpoint selection:

```bash
legonet \
  --dataset-name grapes \
  --network-type per_object_counting \
  --estimate-type keypoints \
  --run-script Inference \
  --val-set Test \
  --have-gt true \
  --weights-mode auto
```

On the first run, LegoNet downloads and verifies the required resized images
and matching checkpoint. Later runs reuse the cached files.

## Known limitations

- Berry points are available for only 111 of the 300 images, and only visually
  countable clusters within those images were annotated with berry points.
- The grape architecture-ablation code and checkpoints are not included in the
  current release.
- Current-code results are subject to software-version differences and original checkpoints availability.

## License and citation

The annotation files in this directory are distributed under the
[Creative Commons Attribution-NonCommercial 4.0 International license
(CC BY-NC 4.0)](LICENSE.txt), separately from LegoNet's BSD-3-Clause source-code
license. The WGISD images remain subject to the WGISD dataset terms and must be
obtained from the upstream project.

When using these files, cite WGISD and the associated LegoNet grape-counting
paper:

> F. Khoroshevsky, S. Khoroshevsky, and A. Bar-Hillel, "Parts-per-Object Count
> in Agricultural Images: Solving Phenotyping Problems via a Single Deep
> Neural Network," *Remote Sensing*, vol. 13, no. 13, article 2496, 2021.
> <https://doi.org/10.3390/rs13132496>

WGISD also provides a permanent dataset record at
<https://doi.org/10.5281/zenodo.3361736>. Follow the upstream WGISD citation
instructions when publishing work based on its images or annotations.
