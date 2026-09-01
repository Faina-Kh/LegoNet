# Experiment outputs

LegoNet writes each run below:

```text
<storage-path>/ExpResults/<dataset>/Results/<current-results-dir>/
```

`current-results-dir` is the run name shown in the GUI or supplied with
`--current-results-dir`. Reusing a name can place artifacts from multiple runs
in the same directory, so use a distinct name when results must be preserved.

## Files produced by inference

```text
<run-directory>/
|-- run_configuration.json
|-- OutputFiles_<split>/
|   `-- with_Vis_results_<split>.txt   # or without_Vis_results_<split>.txt
`-- Vis_<split>/                       # only when drawing is enabled
```

- `run_configuration.json` is the complete machine-readable record of the
  invocation and resolved configuration.
- The results text file contains the concise console log and final evaluation
  summary.
- `detections_data_any_crop.csv` records each evaluated predicted crop.
  `not_found_gt.csv` records unmatched ground-truth objects, and
  `images_without_detections.csv` records images without predictions. These
  per-object CSVs are created only by applicable per-object evaluations.
- `PR_curve_objects.png` is the object-detection precision-recall curve when
  detection ground truth is available.
- Keypoint evaluation may produce `parts_recall_precision.csv` and
  `Points_PR_curve.png`. Protocol-comparison runs add
  `keypoint_protocol_comparison.csv` and protocol-specific plots.

Visualization folders depend on the selected drawing switches. Detection
overviews show full images; per-object folders contain matched-box views,
model crops, or predicted keypoint heatmaps. `GT only` is intended for
annotation inspection. A folder is absent when its corresponding output is
disabled or not applicable to the selected model.

## Training outputs

Training writes `OutputFiles_Train/Train_results.txt` and
`run_configuration.json` in the run directory. Model checkpoints are kept
separately under:

```text
<storage-path>/ExpResults/<dataset>/Weights/
```

This separation lets result directories be renamed or archived without
changing the checkpoint lookup layout.
