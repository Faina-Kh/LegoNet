# Interpreting and reproducing LegoNet experiments

This guide defines the scientific scope of the published comparisons. Dataset
formats, evaluation formulas, and result tables are documented in the
[grapes](datasets/grapes_embrapa-wgisd.md) and
[roots](datasets/roots_grapevine.md) guides. Checkpoint files and provenance are
documented in [pretrained_weights.md](pretrained_weights.md).

## Verification levels

LegoNet provides two different kinds of verification:

1. **CPU software verification** checks installation, CLI configuration,
   dataset handling, checkpoint utilities, evaluation bookkeeping, and result
   generation with fixtures and mocked runtime components. It does not
   reproduce a paper metric.
2. **Scientific comparison** runs a released checkpoint on its documented
   public dataset split and compares current-code output with the corresponding
   paper value where such a comparison is available.

The repository does not currently provide a single manifest-driven command
that reproduces every paper experiment. The dataset guides record the current
reference results and explain where historical data or checkpoints are
unavailable.

## Roots estimation scopes

Roots experiments produce image-level traits in two fundamentally different
ways. Their results must not be combined into one category.

### Direct per-image estimation

The direct models take a complete image as input and estimate total root length
(TRL) without first detecting individual roots. The released direct checkpoints
cover TRL only:

| Experiment | Network type | Estimate type | Released target |
|---|---|---|---|
| Direct TRL, keypoints | `per_image_estimation` | `keypoints` | TRL |
| Direct TRL, regression | `per_image_estimation` | `regression` | TRL |

### Per-root estimation and image aggregation

Per-root models detect roots and estimate length, diameter, and color for
matched individual roots. Image-level values are then calculated from those
per-root outputs:

- image TRL is the sum of per-root length estimates;
- image mean diameter is the mean of per-root diameter estimates; and
- image white fraction is the mean of binary per-root color estimates.

These aggregate results are not direct per-image model results. The roots
dataset guide keeps the two scopes in separate tables.

The standalone per-root keypoint checkpoint has no matching paper Test-set
value. Its color output collapses to the white class, so it is retained as a
documented negative result. The multibranch model is the effective
keypoint-based per-root alternative reported by the paper.

## Grapes per-object counting scope

The active grapes configuration uses only images with contributed berry points
and only grape-cluster boxes containing those points. These are the eligible,
countable clusters; they are not the complete set of original WGISD boxes.

Each predicted box must be IoU-matched to an eligible ground-truth cluster
before its berry count is evaluated. Training, validation, checkpoint
selection, and test evaluation use the full original berry count stored for
that matched ground-truth box. See the grapes dataset guide for filtering,
matching, metrics, and paper comparisons.

## Paper comparisons and split policy

Paper and current-code rows can differ for several reasons:

- roots paper values are five-fold aggregates, while the complete historical
  folds and fold-specific checkpoints are unavailable;
- the released roots files are individual checkpoints;
- the grape detector and keypoint counter use the original experiment
  checkpoints, but software-version differences can slightly change output;
  and
- the released grape regression counter is a newer training run because the
  original MSR checkpoint is unavailable.

A Test result must not be compared with a paper value reported only for the
Validation split. Every reported comparison should identify its dataset,
split, network type, estimate type, checkpoint provenance, and code version.

## Checkpoint selection metrics

Per-object counting minimizes count relative error when selecting the best
checkpoint. Attribute training accepts `--checkpoint-attribute` with `length`,
`diameter`, or `color`; it minimizes relative error for continuous attributes
and classification error rate for color. Length is the default. 1-FVU remains
a reported metric and is not used for checkpoint selection.

## Empty-image policy

Empty-image handling is dataset-specific:

- Roots dataset construction retains images without annotated root objects.
  They provide background examples during detector training, and detections on
  them are false positives during detection evaluation. Per-object attribute
  metrics exclude them.
- Grapes dataset construction removes images without contributed berry points
  and boxes without associated berry points. Excluded images therefore do not
  enter grape detection evaluation.

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
  --draw-individual-object-visualizations true \
  --draw-per-object-estimation-visualizations true \
  --draw-gt-only false
```

The individual-object switch controls matched predicted-box views. The
per-object-estimation switch controls model crops and requested keypoint maps.
The other switches independently control the full-image overview and GT-only
artifacts; `--to-draw` is the master switch.

The detection overview is useful for relatively sparse datasets such as
grapes. Dense roots images may contain many overlapping roots, points, GT
boxes, and predictions, making the full-image overview difficult to read. For
roots, disabling `--draw-detection-overview` while retaining per-object crop
visualizations is usually clearer.

GT-only output is intended for annotation inspection rather than routine model
evaluation. Its folder is not created unless `--draw-gt-only true` is passed.
