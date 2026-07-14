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
across the three splits and contain berry points for the 111 images with the
relevant contributed annotations.
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
└── Datasets/
    └── Embrapa WGISD/
        ├── train.txt
        ├── val.txt
        ├── test.txt
        ├── classes.txt
        └── <WGISD resized JPEG images>
```

Copy the four files from this directory into that runtime directory after
downloading the images. The uploaded annotation files have the following
coverage:

| Split | Images | Images with berry points | Bounding-box rows | Berry-point rows |
|---|---:|---:|---:|---:|
| Train | 218 | 62 | 3,214 | 3,956 |
| Validation | 24 | 24 | 367 | 1,466 |
| Test | 58 | 25 | 850 | 1,927 |
| Total | 300 | 111 | 4,431 | 7,349 |

For the active grape configuration, LegoNet associates each berry point with a
bounding box, removes images without contributed berry points, and removes
bounding boxes containing no contributed berry points. This runtime filtering
produces the berry-annotated subset used by the model.

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
