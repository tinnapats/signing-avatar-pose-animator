# Migration verification — 2026-10-28

Fresh pretrained inference from the migrated archived inputs, 200 clips per run:

| Experiment / model | Correct | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|---:|
| 1 / ST-GCN | 122/200 | 0.6100 | 0.5975 | 0.6100 | 0.601667 |
| 1 / SL-GCN | 96/200 | 0.4800 | 0.4600 | 0.4800 | 0.466667 |
| 2 / ST-GCN, RTMW pixels baseline | 19/200 | 0.0950 | 0.085455 | 0.0950 | 0.0875 |
| 3 / ST-GCN, renderer-assisted | 99/200 | 0.4950 | 0.479167 | 0.4950 | 0.484167 |

These reproduce historical aggregate scores; this is a migration check, not a new independent test or a new full image extraction run.
Experiment 2 here is the baseline stored alongside the Experiment 3 comparison, not the best of every pixel-only trial.

Local result directories under `evaluation/runs/`:
- `20261028-191749-619870`: Experiment 1 ST-GCN
- `20261028-191918-652756`: Experiment 1 SL-GCN
- `20261028-191818-873859`: Experiment 2 baseline
- `20261028-191727-288860`: Experiment 3

Additional checks:
- RTMW actual inference on one archived Avatar PNG: 133 xy points; model and PNG SHA-256 match original metadata. No archive writes.
- Migrated RTMW environment imports rtmlib, onnxruntime and OpenCV.
- Migrated MediaPipe environment imports MediaPipe 0.10.21.
- Main application unit tests: 28 passed. Windows temporary SQLite test required normal filesystem permissions.
- Historical experiments were retained, not all independently re-run. Do not bypass their frozen-production hash checks.

Raw inputs, outputs, weights and environments remain local and are excluded from Git.
