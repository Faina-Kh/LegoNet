# Grapevine roots dataset

LegoNet's `roots` configurations use the **Dataset of Grapevine Roots with
Length, Diameter, and Color Annotations**. The dataset contains in-situ
minirhizotron images with root-object locations and three attributes:

- root length, a regression target;
- root diameter, a regression target; and
- root color, a binary classification target.

These annotations support root detection, direct image-level estimation, and
per-root attribute estimation. Unlike the grapes configuration, roots dataset
construction retains images without annotated root objects.

## Provenance and download

The dataset files are not redistributed in this repository. LegoNet can
download `Grapevines data.zip` from the
[Zenodo dataset record](https://doi.org/10.5281/zenodo.8084106), verify its
published MD5 checksum, and safely extract it into the runtime
`Datasets/Grapevines data/` directory.

Run `legonet-data download roots`, or
allow the normal CLI/GUI workflow to prepare the dataset when it is first
needed.

## Files and annotation format

Each split contains three annotation files alongside the corresponding images:

- `*.csv` supplies image-level data used by direct estimation;
- `*_pointsOutput.csv` supplies point annotations; and
- `*_Dia_Length_Color.txt` supplies per-root locations, identifiers, lengths,
  diameters, and colors.

The JSON-like `*_Dia_Length_Color.txt` files provide per-root point locations
together with `Root_Length`, `Root_Diameter`, and `Root_Color`. Bounding boxes
are derived from each root's annotated points, enlarged by 10 pixels in each
direction, and clipped to the image boundary.

The formats are:

```text
# <Split>.csv (no header)
image.jpg,TRL

# <Split>_pointsOutput.csv (no header)
image.jpg,x1,y1,x2,y2,...
```

The published point file may contain only `image.jpg` on a row. Coordinates
are absolute image pixels. The `*_Dia_Length_Color.txt` file is valid JSON
despite its extension: its top-level object maps record IDs to image records.
An image record uses `processed_name` (legacy data may use `original_name`) and
can include `TRL`, `points`, `roots_num`, and aggregate diameter values. Nested
`root_N` objects contain `Root_Length`, `Root_Diameter`, optional `Root_Color`,
and a flat `points` list `[x1, y1, x2, y2, ...]`.

## Runtime layout

Place the downloaded dataset under the configured LegoNet storage directory:

```text
<storage-path>/
`-- Datasets/
    `-- Grapevines data/
        |-- sub_Train/
        |   |-- Train.csv
        |   |-- Train_pointsOutput.csv
        |   `-- Train_Dia_Length_Color.txt
        |-- sub_Val/
        |   |-- Val.csv
        |   |-- Val_pointsOutput.csv
        |   `-- Val_Dia_Length_Color.txt
        `-- sub_Test/
            |-- Test.csv
            |-- Test_pointsOutput.csv
            `-- Test_Dia_Length_Color.txt
```

Annotation files reference the corresponding image locations within the
downloaded dataset.

## Attribute encoding

TRL and diameter are continuous regression targets. Their units and annotation
protocol follow the dataset documentation.

Color is loaded as a binary classification target:

| Stored annotation | LegoNet label |
|---|---:|
| `White` | 1 |
| Any other annotated color | 0 |
| Missing color annotation | -1 (excluded from color metrics) |

Reports refer to labels `1` and `0` as `white` and `non_white`, respectively.

## Supported configurations

| Network type | Estimate type | Output |
|---|---|---|
| `bbox_detection` | Not used | Root bounding boxes |
| `per_image_estimation` | `keypoints` or `regression` | Attribute estimated directly from the complete image |
| `per_object_attributes` | `keypoints` or `regression` | Length, diameter, and color for each matched root |
| `per_object_attributes_multibranch` | `keypoints` | Per-root attributes using the multibranch keypoint architecture |

The released direct per-image checkpoints estimate total root length (TRL).
Per-root models estimate length, diameter, and color; their outputs can also be
aggregated into image-level summaries as described below.

## Evaluation levels

Attribute evaluation is reported at two distinct levels. They should not be
mixed because they answer different questions.

### Bounding-box metrics

Bounding-box detections are evaluated after confidence filtering and
non-maximum suppression. A prediction is a true positive when it has at least
the configured intersection over union (IoU) with a ground-truth root box and
that box has not already been matched to a higher-scoring prediction. An
unmatched prediction or a duplicate prediction for an already matched box is a
false positive. A ground-truth root box with no matching prediction is a false
negative.

```text
precision = true positives / (true positives + false positives)
recall    = true positives / (true positives + false negatives)
```

Precision therefore measures how many predicted root boxes are correct, while
recall measures how many annotated roots are detected. Images without
annotated roots remain in roots detection evaluation; every prediction on such
an image is a false positive. The attribute metrics below use only predictions
that are matched to usable per-root annotations.

### Per-object evaluation

A predicted object is evaluated against an attribute target only after its
bounding box is matched to an unclaimed ground-truth box at the configured IoU
threshold. Predictions without a matched attribute annotation are excluded
from attribute metrics and remain part of the separate detection evaluation.

For TRL and diameter, LegoNet reports mean absolute error (MAE), mean squared
error (MSE), Mean Relative Deviation (MRD), and 1 minus the fraction of
variance unexplained (1-FVU). As in the grapes evaluation, MRD is the mean
absolute error relative to the corresponding nonzero ground-truth value.

For color, LegoNet reports classification accuracy and error rate:

```text
accuracy   = correctly classified matched objects / evaluated matched objects
error_rate = 1 - accuracy
```

The report also includes balanced accuracy, macro precision, macro recall,
macro F1, the confusion matrix, evaluation coverage, and binary-label 1-FVU.
Coverage is reported because color accuracy alone does not describe how many
eligible objects were successfully detected and evaluated.

### Per-image evaluation

Per-image evaluation first reduces the object values in each image to one GT
value and one predicted value:

```text
image TRL       = sum of the image's per-object root lengths
image diameter  = mean of the image's per-object root diameters
image color     = mean of the image's binary per-object color labels
```

Consequently, mean image color represents the fraction of evaluated roots
classified as white. For example, a mean color of `0.75` means that 75% of the
image's evaluated roots have label `white`.

Error is calculated between corresponding image-level GT and predicted values.
For TRL and diameter, the relative deviation for image `i` is:

```text
relative_deviation_i = abs(gt_i - prediction_i) / gt_i
MRD = mean(relative_deviation_i)
```

Images whose GT aggregate is zero do not enter the MRD value.
For diameter, the same summary line also reports the mean absolute difference
between image-level GT and predicted mean diameters.

For color, only absolute error is used:

```text
color_absolute_error_i = abs(mean_gt_color_i - mean_predicted_color_i)
```

No image-level relative color error is calculated.

Per-image 1-FVU is then calculated **once across the vectors of image-level
values**. It is not calculated independently inside every image and those
scores are not averaged:

```text
MSE = mean((image_gt - image_prediction) ** 2)
1-FVU = 1 - MSE / variance(image_gt)
```

This calculation is performed separately for image-level length sums (TRLs), diameter
means, and color means. If the GT image-level values have zero variance,
1-FVU is undefined and is reported as `n/a`. Each attribute's 1-FVU is printed
beside that attribute's mean image-level error rather than in a separate
combined summary.

### Detection scope

Roots models support training and inference on images without annotated root
objects. Such images provide background examples during detector training, and
bounding-box detection evaluates them during inference. Any detections on
these images count as false positives. Per-object attribute evaluation requires
usable matched attribute annotations and therefore excludes these images. The
two evaluations intentionally operate on different scopes.

### Point-localization interpretation

Point-localization AP is reported for the intermediate heatmaps of all
keypoint-based configurations for consistency. For the standalone per-root
keypoint model, point-localization performance is relatively low despite good
final root-attribute estimation performance. Point-detection metrics for this
configuration were not reported in the original paper.

Per-root point AP is evaluated only on non-empty detector-produced crops that
were matched one-to-one to a ground-truth root box at the configured IoU
threshold. Unmatched or duplicate detections remain part of bounding-box
evaluation but are excluded from point AP, allowing the intermediate point
detector to be assessed conditional on successful root-box detection.

Since the final attributes are obtained through learned downstream estimation
layers, intermediate point-localization performance and final attribute
estimation measure different aspects of the model. We report both for
transparency; the multibranch model remains the paper-aligned keypoint-based
per-root configuration. The PCK matching rule and AP protocol are defined in
the [reproduction guide](../reproduction.md#point-localization-evaluation).

## Paper comparison

The paper rows below report the published aggregate results. The current-code
rows report evaluation of the corresponding released checkpoint. Because the
historical folds and complete fold-specific checkpoint set are unavailable,
the current-code rows are reference results rather than exact reproductions of
the paper aggregates. Paper values are from the
[grapevine-root attribute study](https://doi.org/10.1016/j.compag.2024.109457).

### Direct per-image TRL

These models estimate TRL directly from the complete image; their results are
not obtained by aggregating detected-root predictions.

| Estimate type | Source | TRL MRD | 1-FVU |
|---|---|---:|---:|
| Keypoints | Paper | 16.0% | 0.92 |
| Keypoints | Current code | 14.4% | 0.93 |
| Regression | Paper | 13.7% | 0.92 |
| Regression | Current code | 16.7% | 0.93 |

### Per-object attributes — regression

Per-object results:

| Source | Color error | Color 1-FVU | Length MRD | Length 1-FVU | Diameter MRD | Diameter 1-FVU | Diameter absolute difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| Paper | 8.8% | 0.63 | 15.5% | 0.91 | 23.5% | 0.60 | 0.086 |
| Current code | 7.7% | 0.67 | 17.2% | 0.89 | 22.5% | 0.59 | 0.086 |

Per-image aggregation of those per-root predictions:

| Source | White-fraction error | White-fraction 1-FVU | TRL MRD | TRL 1-FVU | Mean-diameter MRD | Mean-diameter 1-FVU | Mean-diameter absolute difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| Paper | 11.5% | 0.69 | 38.9% | 0.40 | 17.6% | 0.42 | 0.070 |
| Current code | 11.2% | 0.75 | 40.4% | 0.337 | 17.9% | 0.42 | 0.070 |

### Per-object attributes — standalone keypoints

No matching paper Test-set values are available for this standalone
configuration.

| Level | Source | Color error | Color 1-FVU | Length/TRL MRD | Length/TRL 1-FVU | Diameter MRD | Diameter 1-FVU | Diameter absolute difference |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Per object | Current code | 61.0% | -1.565 | 15.4% | 0.91 | 27.6% | 0.52 | 0.097 |
| Per-image aggregation | Current code | 45.14% | -1.5322 | 39.1% | 0.3915 | 20.6% | 0.358 | 0.076 |

The released standalone keypoint checkpoint predicts every color output as
white. This failed color behavior motivated the multibranch alternative; the
negative color 1-FVU values indicate performance worse than predicting the
mean target.

### Per-object attributes — multibranch keypoints

Per-object results:

| Source | Color error | Color 1-FVU | Length MRD | Length 1-FVU | Diameter MRD | Diameter 1-FVU | Diameter absolute difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| Paper | 9.1% | 0.62 | 14.9% | 0.92 | 25.0% | 0.52 | 0.095 |
| Current code | 7.7% | 0.67 | 15.4% | 0.91 | 25.3% | 0.50 | 0.097 |

Per-image aggregation of those per-root predictions:

| Source | White-fraction error | White-fraction 1-FVU | TRL MRD | TRL 1-FVU | Mean-diameter MRD | Mean-diameter 1-FVU | Mean-diameter absolute difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| Paper | 16.5% | 0.46 | 38.6% | 0.42 | 22.1% | 0.12 | 0.090 |
| Current code | 16.4% | 0.46 | 39.1% | 0.39 | 22.1% | 0.12 | 0.090 |


## Visualization guidance

Roots images are dense: a single image can contain many overlapping root
objects, point annotations, GT boxes, and predicted boxes. Consequently, a
full-image detection overview can become difficult to interpret. Per-object
matched-box images and predicted crops are generally more useful for checking
root attribute estimates.

Use the master drawing switch together with the overview controls:

```text
--to-draw true
--draw-detection-overview false
--draw-gt-only false
```

- `--draw-detection-overview` controls the full-image GT/prediction overlay.
  It defaults to `true` for compatibility and can be disabled without turning
  off per-object visualizations.
- `--draw-gt-only` controls annotation-debugging images containing GT points
  and boxes without predictions. It defaults to `false`, and the `GT only`
  folder is created only when this option is enabled.
- Per-object predicted-box images, predicted crops, and requested keypoint
  maps remain controlled by `--to-draw`.

The GT box drawn for each matched prediction is selected by its stored
`bbox_id`, not by its row position in the annotation array. This is important
for the roots dataset because IDs are zero-based and may not remain contiguous
after invalid boxes are filtered.

## Example

Run roots per-object attribute inference on the public test split with
automatic dataset and checkpoint selection:

```bash
legonet \
  --dataset-name roots \
  --network-type per_object_attributes \
  --estimate-type keypoints \
  --run-script Inference \
  --val-set Test \
  --have-gt true \
  --to-draw true \
  --draw-detection-overview false \
  --draw-gt-only false \
  --weights-mode auto
```

On the first run, LegoNet downloads and verifies the dataset and matching
checkpoint. Later runs reuse the cached files.

## Known limitations

- The associated paper reports five-fold results, but the historical fold
  definitions and complete fold-specific checkpoint set are unavailable.
  Consequently, the released individual checkpoints cannot reproduce the
  paper's five-fold aggregates exactly.
- Released direct per-image checkpoints cover TRL only.
- The standalone per-root keypoint checkpoint's Test-set color output
  collapses to the white class; the multibranch model is the effective
  keypoint-based per-root alternative documented by the paper.

## License and citation

The dataset is distributed under the [Creative Commons Attribution 4.0
International license](https://creativecommons.org/licenses/by/4.0/). When
using it, cite the dataset record and the associated publication:

> Faina Khoroshevsky, Kaining Zhou, and Naftali Lazarovitch. (2023).
> *Dataset of Grapevine roots with length, diameter, and color annotations*
> [Dataset]. Zenodo. <https://doi.org/10.5281/zenodo.8084106>
>
> F. Khoroshevsky, K. Zhou, A. Bar-Hillel, O. Hadar, S. Rachmilevitch,
> J. E. Ephrath, N. Lazarovitch, and Y. Edan, "A CNN-based framework for
> estimation of root length, diameter, and color from in situ minirhizotron
> images," *Computers and Electronics in Agriculture*, vol. 227, article
> 109457, 2024. <https://doi.org/10.1016/j.compag.2024.109457>
