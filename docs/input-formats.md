# Dataset input formats

LegoNet currently supports its two published dataset layouts rather than a
generic image-upload format. Dataset paths are resolved below
`<storage-path>/Datasets/`; filenames in annotation files must identify the
corresponding images.

## Grapes

The Embrapa WGISD integration uses comma-separated rows without a header:

```text
image.jpg,grapes,x1,y1,x2,y2
image.jpg,grapes,x,y
```

The first form is a grape-cluster bounding box and the second is a berry
point. Coordinates are absolute pixels in the resized WGISD image; box corners
are `(x1, y1)` and `(x2, y2)`. `classes.txt` contains `grapes,0`.
See [Embrapa WGISD grape dataset](datasets/grapes_embrapa-wgisd.md) for the
layout, filtering rules, and provenance.

## Grapevine roots

Each Train, Val, or Test split uses three complementary files:

- `<Split>.csv`: headerless `image_filename,TRL` rows for direct image-level
  estimation.
- `<Split>_pointsOutput.csv`: image filenames with optional flat `x,y`
  coordinate pairs; the published data may contain filename-only rows.
- `<Split>_Dia_Length_Color.txt`: JSON content (despite the `.txt` suffix)
  containing image records and nested `root_N` records. Point arrays are flat
  absolute-pixel lists: `[x1, y1, x2, y2, ...]`.

Per-root records provide `Root_Length`, `Root_Diameter`, optional
`Root_Color`, and `points`. LegoNet derives a box from each point set, expands
it by 10 pixels, and clips it to the image. See
[Grapevine roots dataset](datasets/roots_grapevine.md) for the complete schema
and label encoding.
