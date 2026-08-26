# Pretrained checkpoint status

LegoNet does not currently provide checkpoints that reproduce every result or
architecture ablation reported in the associated papers. 

### Split a full checkpoint into partial weights

Use the standalone converter for per-object checkpoints saved by the current
code. It validates the selected architecture before creating separate detector
and per-object head checkpoints:

```powershell
python scripts/split_full_checkpoint.py `
  --full-weights-file "C:\weights\legonet_epoch=120.pt" `
  --network-type per_object_counting `
  --estimate-type regression `
  --detector-output-file "C:\weights\legonet_bbox_grapes.pt" `
  --per-object-output-file "C:\weights\legonet_counting_reg.pt"
```

For an attributes head, pass its attribute names. Each name maps to an
`estimator_<name>` module:

```powershell
python scripts/split_full_checkpoint.py `
  --full-weights-file "C:\weights\legonet_epoch=90.pt" `
  --network-type per_object_attributes `
  --estimate-type keypoints `
  --attribute-names length diameter color `
  --detector-output-file "C:\weights\legonet_bbox_roots.pt" `
  --per-object-output-file "C:\weights\legonet_attributes_kp.pt"
```

Leave `--attribute-names` out when the head uses the single module name
`estimator`. Existing files are preserved unless `--overwrite` is supplied.
The converter refuses architecture mismatches and never overwrites its source
checkpoint. `--detector-output-file` is optional; omit it when the detector
weights have already been saved and only a new per-object head file is needed.

The same operation is available in a separate Streamlit page:

```powershell
streamlit run apps/checkpoint_splitter.py
```

The converter page accepts an output directory separately from the detector
and per-object `.pt` filenames. When Streamlit runs locally, the output
directory can also be selected with the native folder browser.

To perform the reverse operation, combine compatible detector and per-object
partial checkpoints into one full checkpoint:

```powershell
python scripts/combine_partial_checkpoints.py `
  --detector-weights-file "C:\weights\legonet_bbox_grapes.pt" `
  --per-object-weights-file "C:\weights\legonet_counting_reg.pt" `
  --network-type per_object_counting `
  --estimate-type regression `
  --full-output-file "C:\weights\legonet_counting_reg_full.pt"
```

The combiner validates both module sets, adds the `bbox_detection.` prefix to
detector keys, verifies the saved checkpoint, and refuses to overwrite either
source file. Named attributes use the same `--attribute-names` option as the
split operation. Combination is rejected whenever either partial checkpoint's
modules do not exactly match the modules defined by the selected network type,
estimate type, and attribute names.

The reverse operation also has a local Streamlit page:

```powershell
streamlit run apps/checkpoint_combiner.py
```

To remove one obsolete module, such as `find_2`, from a checkpoint through a
local Streamlit page, run:

```powershell
streamlit run apps/checkpoint_module_remover.py
```

The page preserves the source checkpoint, displays the modules before and
after removal, and writes the cleaned weights to a separate output file.

Boolean values accept `true`, `false`, `yes`, `no`, `1`, or `0`. Explicit
false values are preserved.

Checkpoint loading and legacy checkpoint export are mutually exclusive:

```text
--load-weights true
--save-from-model-file true
```

cannot be used together.





## Grapes (Embrapa WGISD)

The grape detection and keypoint-based per-object counting checkpoints are
believed to be the checkpoints used for the reported experiments, or to produce
very similar results. This has not yet been verified by rerunning the complete
evaluation with the public repository.

The available regression-based per-object counting checkpoint was produced by
a later training run. It is not the original checkpoint used for the paper, and
its results should not be presented as reproducing the reported regression
results. The latest comparison results are currently stored on an unavailable
remote machine and must be checked later.

| Configuration | Candidate file | Size (MiB) | SHA-256 | Status |
|---|---|---:|---|---|
| Bounding-box detection | `legonet_bbox_grapes.pt` | 138.90 | `14586c6ff25938e4ceb29d06cab10701c1da0a80065adef83a5469fbf4c44837` | Believed paper-aligned; verification pending |
| Per-object counting, keypoints | `legonet_CountWithKeyPoints.pt` | 129.60 | `94de002bf7d6e35aa2fc043bc3f1cf504b0856343df0a36c2a1bc37dd70ae183` | Believed paper-aligned; verification pending |
| Per-object counting, regression | `legonet_epoch=59.pt` | 273.19 | `01cb7b39da761cd13fdb8a734ce17d2512ce22f34013f576c29d7c1bea26d777` | Later training; not the paper checkpoint |

The architecture-ablation implementations and their checkpoints are not part
of the current codebase. Reconstructing them may be possible from older code,
but doing so requires additional development and validation and is outside the
current release scope.

## Grapevine roots

The available root checkpoints correspond to model configurations relevant to
the later *Computers and Electronics in Agriculture* paper. However, the paper
reports results aggregated over five-fold validation. The historical fold
definitions and the complete set of fold-specific checkpoints are not
available, so these files cannot reproduce the reported five-fold results
exactly. Per-image and per-object outputs from an individual available
checkpoint may differ from the paper's aggregated results.

| Configuration | Candidate file | Size (MiB) | SHA-256 | Status |
|---|---|---:|---|---|
| Bounding-box detection | `legonet_bbox_roots.pt` | 138.90 | `939054168e9f6746163c0c0b9905b3fa02889e8c9cf4333f7d38392501da9009` | Relevant checkpoint; five-fold reproduction unavailable |
| Per-image estimation, keypoints | `legonet_TRLwithKeyPoints.pt` | 129.59 | `f8759c40712a7e7f0637b24513d1a0f2831d6cc579acf3c5732b4f730472d568` | Relevant checkpoint; five-fold reproduction unavailable |
| Per-image estimation, regression | `legonet_TRLwithReg.pt` | 125.15 | `5b13f23b07c2f48ae8fbbc4f0048611d2c878d6f655e21e1f2e6495b16a9cb08` | Relevant checkpoint; five-fold reproduction unavailable |
| Per-object attributes, keypoints | `legonet_AttrWithKeyPoints.pt` | 131.90 | `3e927fcd988d9f61205ea17e7046a1442802eb2e5b812d2dada56f66f4b2383f` | Relevant checkpoint; five-fold reproduction unavailable |
| Per-object attributes, regression | `legonet_AttrWithReg.pt` | 143.62 | `d8d38bb163e8e1d5b09c5e99ecf03847ccf96c4f5ceb188529a29bb85e5f903e` | Relevant checkpoint; five-fold reproduction unavailable |
| Per-object attributes, multibranch keypoints | `legonet_AttrWith2B2F.pt` | 261.53 | `132aeeb0c5341a6c73c936aaf29364b87ecc0c6ae55220ea76e028bf8953243a` | Relevant checkpoint; five-fold reproduction unavailable |

## Checkpoint verification

Before a checkpoint is described as a verified, supported download:

1. Test that it loads into the matching model configuration in the public code.
2. Run inference on the documented public dataset setup.
3. Record the exact code version, configuration, and evaluation output.
4. Upload the verified file to a versioned research archive such as Zenodo.
5. Add its permanent URL and checksum to a machine-readable download manifest.

Until those checks are complete, files marked as candidates are an inventory
of research artifacts rather than a guarantee of exact result reproduction.

## Checkpoint utilities

LegoNet includes validated utilities for converting between full-model and
modular checkpoints. Split a per-object full checkpoint with:

```powershell
python scripts/split_full_checkpoint.py `
  --full-weights-file "C:\weights\legonet_epoch=120.pt" `
  --network-type per_object_counting `
  --estimate-type regression `
  --detector-output-file "C:\weights\legonet_bbox_grapes.pt" `
  --per-object-output-file "C:\weights\legonet_counting_reg.pt"
```

For an attributes checkpoint, add `--attribute-names length diameter color`.
The detector output is optional when only the per-object head is required.
Existing output files are preserved unless `--overwrite` is supplied, and the
source checkpoint is never overwritten.

Combine compatible partial checkpoints with:

```powershell
python scripts/combine_partial_checkpoints.py `
  --detector-weights-file "C:\weights\legonet_bbox_grapes.pt" `
  --per-object-weights-file "C:\weights\legonet_counting_reg.pt" `
  --network-type per_object_counting `
  --estimate-type regression `
  --full-output-file "C:\weights\legonet_counting_reg_full.pt"
```

Both utilities validate that checkpoint modules match the selected network and
estimator architectures. Equivalent local Streamlit tools are available with:

```powershell
streamlit run apps/checkpoint_splitter.py
streamlit run apps/checkpoint_combiner.py
streamlit run apps/checkpoint_module_remover.py
```

The module remover preserves its source and writes the cleaned checkpoint to a
separate file.
