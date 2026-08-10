# Grapevine roots dataset

LegoNet's `roots` configurations use the **Dataset of Grapevine Roots with
Length, Diameter, and Color Annotations**. The dataset contains in-situ
minirhizotron images with root-object locations and three attributes:

- total root length (TRL), a regression target;
- root diameter, a regression target; and
- root color, a binary classification target.

The dataset files are not redistributed in this repository. Download them
from the [Zenodo dataset record](https://doi.org/10.5281/zenodo.8084106).

## License and citation

The dataset is distributed under the [Creative Commons Attribution 4.0
International license](https://creativecommons.org/licenses/by/4.0/). When
using it, cite the dataset record and the associated publication:

> F. Khoroshevsky, K. Zhou, A. Bar-Hillel, O. Hadar, S. Rachmilevitch,
> J. E. Ephrath, N. Lazarovitch, and Y. Edan, "A CNN-based framework for
> estimation of root length, diameter, and color from in situ minirhizotron
> images," *Computers and Electronics in Agriculture*, vol. 227, article
> 109457, 2024. <https://doi.org/10.1016/j.compag.2024.109457>

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

Annotation files reference the corresponding image locations in the downloaded
dataset. The JSON-like `*_Dia_Length_Color.txt` files provide per-root point
locations together with `Root_Length`, `Root_Diameter`, and `Root_Color`.
Bounding boxes are derived from each root's annotated points and enlarged by
10 pixels in each direction, clipped to the image boundary.

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

## Evaluation levels

Attribute evaluation is reported at two distinct levels. They should not be
mixed because they answer different questions.

### Per-object evaluation

A predicted object is evaluated against an attribute target only after its
bounding box is matched to an unclaimed ground-truth box at the configured IoU
threshold. Predictions without a matched attribute annotation are excluded
from attribute metrics and remain part of the separate detection evaluation.

For TRL and diameter, LegoNet reports mean absolute error (MAE), mean squared
error (MSE), mean relative error, and 1 minus the fraction of variance
unexplained (1-FVU).

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
For TRL and diameter, the relative error for image `i` is:

```text
relative_error_i = abs(gt_i - prediction_i) / gt_i
```

Images whose GT aggregate is zero do not enter the mean relative-error value.
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

This calculation is performed separately for image-level TRL sums, diameter
means, and color means. If the GT image-level values have zero variance,
1-FVU is undefined and is reported as `n/a`. Each attribute's 1-FVU is printed
beside that attribute's mean image-level error rather than in a separate
combined summary.

## Detection scope

Bounding-box detection evaluates every image, including images with no
ground-truth boxes; predictions on empty images count as false positives.
Per-object attribute evaluation requires usable matched attribute annotations.
The two evaluations therefore intentionally operate on different scopes.

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

Run roots per-object attribute inference with:

```bash
python scripts/run_legonet.py \
  --storage-path /path/to/legonet-storage \
  --dataset-name roots \
  --network-type per_object_attributes \
  --estimate-type withKeyPoints \
  --run-script Inference \
  --val-set Test \
  --have-gt true \
  --to-draw true \
  --draw-detection-overview false \
  --draw-gt-only false \
  --weights-mode full \
  --full-weights-file /path/to/per_object_attributes.pt
```
