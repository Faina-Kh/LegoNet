# LegoNet grape annotations

This directory contains the processed annotation files used by LegoNet's grape
experiments. The corresponding images are not redistributed here.

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

## Files and format

The expected files are:

```text
train.txt
val.txt
test.txt
classes.txt
```

Each annotation row is comma-separated and uses absolute pixel coordinates:

```text
image.jpg,grapes,x1,y1,x2,y2  # grape-cluster bounding box
image.jpg,grapes,x,y           # berry-point annotation
```

`classes.txt` maps the `grapes` class name to its numeric class identifier.
LegoNet uses the four-column and six-column row lengths to distinguish points
from bounding boxes.

## Image setup

Download the resized JPEG images from WGISD's
[`data/` directory](https://github.com/thsant/wgisd/tree/master/data). Do not use
the images from WGISD's `original_resolution/` directory with these coordinates.

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

Copy the four files from this directory into that runtime directory after
downloading the images. The uploaded annotation files have the following
coverage:

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

## Detection reproducibility note

The grape bounding-box detector is evaluated with a minimum confidence score
of `0.7`, an IoU threshold of `0.5`, and an NMS threshold of `0.3`. In the
current PyTorch/Torchvision environment, evaluation of the available detector
checkpoint on the test set produced 27 matched GT boxes among 57 detections:
47.4% precision. The recall agreed with the value reported in the paper, while
the paper's reported precision was approximately 49%.

Two current false-positive detections have confidence scores immediately above
the `0.7` boundary. A threshold-sensitivity check at `0.713` removed those two
detections without removing any true positives, giving 27 matches among 55
detections, or 49.1% precision, while leaving recall unchanged. This indicates
that the precision difference is caused by detections close to the confidence
threshold, plausibly due to numerical or NMS differences between the historical
and current package versions. The documented evaluation threshold remains
`0.7`; the `0.713` result is reported only as a reproducibility diagnostic.

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
