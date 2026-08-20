# Pretrained checkpoint status

LegoNet does not currently provide checkpoints that reproduce every result or
architecture ablation reported in the associated papers. The files listed here
are candidate release checkpoints that still require compatibility and
evaluation checks against the public code and datasets.

No claim of exact paper-result reproduction should be inferred unless a
checkpoint is explicitly marked as verified in a future release.

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
