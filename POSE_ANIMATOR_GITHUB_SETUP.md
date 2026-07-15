# Pose Animator (GitHub) with Your Dataset

This setup uses the official `pose-animator` runtime classes from GitHub:
- `illustrationGen/skeleton.js`
- `illustrationGen/illustration.js`

## Fast path (old workflow: type text in UI)

Run:

```powershell
cd C:\pro1end
.\.venv\Scripts\python.exe .\run_pose_animator_server.py --port 8025
```

Open:

`http://127.0.0.1:8025/dataset_player.html`

Then type text (example: `hello`) and click `Generate From Text`.

## Manual path: export your CSV sequence to JSON

Run from `D:\project_1`:

```powershell
python .\export_pose_animator_sequence.py `
  --data-dir ".\SLclean" `
  --files "SLclean\a.csv" `
  --output ".\pose-animator\resources\data\my_sequence.json"
```

Text lookup example:

```powershell
python .\export_pose_animator_sequence.py `
  --data-dir ".\SLclean" `
  --text "hello" `
  --output ".\pose-animator\resources\data\hello_sequence.json"
```

## 2) Run local web server (manual JSON mode)

```powershell
cd .\pose-animator
python -m http.server 8025
```

Open:

`http://127.0.0.1:8025/dataset_player.html`

## 3) In the player page

1. Load your JSON file (`resources/data/my_sequence.json` or any exported file).
2. Choose built-in avatar or upload/drop your own SVG.
3. Press `Play`.

## Notes

- This is the GitHub pose-animator pipeline (browser + Paper.js + skeleton rig).
- It uses pose + face landmarks (same as original pose-animator design).
- Hand landmarks from your CSV are not part of the original pose-animator skeleton and are not animated directly.
- `parcel` build scripts in this old repo can fail on modern Node versions (e.g., Node 22), so `dataset_player.html` is configured to run directly via ES modules and a local HTTP server.
