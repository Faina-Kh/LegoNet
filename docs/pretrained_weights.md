# Pretrained weights

LegoNet publishes checkpoints for every configuration listed in the main
README's supported-configuration table. The files are hosted in the
[LegoNet Zenodo record](https://doi.org/10.5281/zenodo.21966953), selected by
the CLI or Streamlit GUI, verified against the MD5 values recorded in
`src/legonet/pretrained.py`, and cached under
`<storage-path>/checkpoints/zenodo-21966953/`.

These checkpoints cover the supported public examples, but not every
architecture ablation or historical fold reported in the associated papers.
Paper comparisons and their limitations are documented in the
[dataset guides](datasets/) and [reproduction guide](reproduction.md).

## Grapes — Embrapa WGISD

The detector and keypoint-based counter are the original checkpoints used for
the published experiments. The regression counter was produced by a newer
training run because the original MSR checkpoint was unavailable.

| Component | Published file | Size (MiB) | MD5 | Provenance |
|---|---|---:|---|---|
| Detector | `legonet_bbox_grapes.pt` | 138.90 | `ebd63a96df0cd4e39e3d70fe4d3f1601` | Original experiment checkpoint |
| Keypoint counter, partial | `legonet_partial_counting_keypoints_grapes.pt` | 129.60 | `c95a3684ba321c23b0eb348b516d4406` | Original experiment checkpoint |
| Keypoint counting model, full | `legonet_full_counting_keypoints_grapes.pt` | 268.52 | `af0a89b5371d5b1ea58389bf39b9ecd4` | Composed from the published detector and keypoint counter |
| Regression counter, partial | `legonet_partial_counting_regression_grapes.pt` | 125.16 | `f2faab2f869bf1a1acd014bdd1b8923f` | Newer training run; not the paper checkpoint |
| Regression counting model, full | `legonet_full_counting_regression_grapes.pt` | 264.08 | `e28aadac49fe027b4441564c65c5c688` | Composed using the newer regression counter |

The original paper also reports architecture ablations whose implementations
and checkpoints are not part of the current release.

## Grapevine roots

The roots paper reports results aggregated over five folds. The historical
fold definitions and complete fold-specific checkpoint set are unavailable,
so the published individual checkpoints cannot reproduce those five-fold
aggregates exactly.

| Component | Published file | Size (MiB) | MD5 |
|---|---|---:|---|
| Detector | `legonet_bbox_roots.pt` | 138.90 | `b2de6847c6e50d6026b80968ec9be52b` |
| Direct TRL, keypoints | `legonet_direct_TRL_keypoints_roots.pt` | 129.59 | `ec846449ce1e45df3883b67b626ee269` |
| Direct TRL, regression | `legonet_direct_TRL_regression_roots.pt` | 125.15 | `dc69814f45a31937ec78fe2752fd69f1` |
| Per-root attributes, keypoints, partial | `legonet_partial_attributes_keypoints_roots.pt` | 131.90 | `4982c1004466612a7cd2c58418203ca8` |
| Per-root attributes, keypoints, full | `legonet_full_attributes_keypoints_roots.pt` | 270.83 | `dea53894ba64d407b645609860490679` |
| Per-root attributes, regression, partial | `legonet_partial_attributes_regression_roots.pt` | 134.52 | `24c66522ba0efbc8cf5b93446e613db5` |
| Per-root attributes, regression, full | `legonet_full_attributes_regression_roots.pt` | 273.46 | `4d9c168757d4a5fc8f100571eeb6d08c` |
| Multibranch attributes, keypoints, partial | `legonet_partial_attributes_multibranch_roots.pt` | 261.53 | `1ed96a108fb64ffed0f044ffd3ea26a5` |
| Multibranch attributes, keypoints, full | `legonet_full_attributes_multibranch_roots.pt` | 400.50 | `b2dad2d63cafa451ab31d9d879c74303` |

## Automatic and manual loading

Inference defaults to `--weights-mode auto`, which downloads or reuses the
matching full-model checkpoint. Use `--weights-mode full`, `partial`,
`detector_only`, or `none` for explicit loading workflows; the required path
options are summarized in the main README.

Automatic selection is configuration-aware. If no published checkpoint exists
for a requested combination, LegoNet fails with a clear message instead of
silently loading an incompatible file.

## Checkpoint utilities

LegoNet includes validated utilities for converting between full-model and
modular checkpoints. Split a per-object full checkpoint with:

```powershell
python scripts/split_full_checkpoint.py `
  --full-weights-file "C:\weights\legonet_full_counting_regression_grapes.pt" `
  --network-type per_object_counting `
  --estimate-type regression `
  --detector-output-file "C:\weights\legonet_bbox_grapes.pt" `
  --per-object-output-file "C:\weights\legonet_partial_counting_regression_grapes.pt"
```

For an attributes checkpoint, add `--attribute-names length diameter color`.
The detector output is optional when only the per-object head is required.
Existing files are preserved unless `--overwrite` is supplied, and the source
checkpoint is never overwritten.

Combine compatible partial checkpoints with:

```powershell
python scripts/combine_partial_checkpoints.py `
  --detector-weights-file "C:\weights\legonet_bbox_grapes.pt" `
  --per-object-weights-file "C:\weights\legonet_partial_counting_regression_grapes.pt" `
  --network-type per_object_counting `
  --estimate-type regression `
  --full-output-file "C:\weights\legonet_full_counting_regression_grapes.pt"
```

Both utilities validate the checkpoint modules against the selected network
and estimator architecture and refuse unsafe overwrites. Equivalent local
Streamlit tools are available with:

```powershell
streamlit run apps/checkpoint_splitter.py
streamlit run apps/checkpoint_combiner.py
streamlit run apps/checkpoint_module_remover.py
```

The module remover preserves its source checkpoint and writes the cleaned
version to a separate file.
