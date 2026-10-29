# Coordinate-fidelity evaluator

This is a 2D hand-coordinate audit, separate from ST-GCN/SL-GCN word recognition.
PCK measures the fraction of points within a distance threshold; it is not ASL accuracy.
RMSE measures geometric error, not human comprehension.

## Source files (repository root)

- `measure_render_fidelity.py`: prepare six sequences from local SQLite, then summarize original capture output.
- `measure_render_fidelity.mjs`: instrument a temporary renderer build and capture endpoints.
- `verify_render_fidelity.mjs`: additional camera-transform assertions and Canvas checksums/screenshots.
- `summarize_fidelity_verification.py`: summarize verification captures and record source/input hashes.

## Current checkout workflow

Requires local `SLclean.sqlite3`, Python with numpy/pandas, Node.js, Vite, Playwright and Chrome.
Install Vite and Playwright under `frontend-tooling/node_modules` (see FRIEND_INSTALL_TH.md). Chromium installed by Playwright is the default. Optional overrides: `AUDIT_VITE`, `AUDIT_PLAYWRIGHT`, `AUDIT_BROWSER`.

Run from the repository root (use the installed Python environment):

```powershell
python measure_render_fidelity.py prepare
$env:FIDELITY_SOURCE = (Get-Location).Path
$env:FIDELITY_OUTPUT = Join-Path (Get-Location).Path ('fidelity_verification/run-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
node verify_render_fidelity.mjs
python summarize_fidelity_verification.py
```

The original capture workflow is `python measure_render_fidelity.py prepare`, then `node measure_render_fidelity.mjs`, then `python measure_render_fidelity.py summarize`. Its files are written to `fidelity_results`.

## Preserve historical evidence

Without overrides, the verification pair now uses this checkout and writes to `fidelity_verification/current`, keeping historical results at the parent level untouched. Do not interpret historical results as newly evaluated newton results.
Use the same `FIDELITY_SOURCE` and `FIDELITY_OUTPUT` for capture and summarization; do not modify the source or inputs between them. Always choose a fresh output directory for new verification runs.
The preparation command overwrites `fidelity_results/inputs.json`; preserve any prior results before preparing again. The renderer hooks deliberately fail if the expected source layout changes; do not bypass those checks.

Images, generated JSON, reports, SQLite, model weights and dependencies are not published with this code. No new fidelity scores are claimed merely by uploading these scripts.
