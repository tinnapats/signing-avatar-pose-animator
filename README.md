# Signing Avatar / Pose Animator

A local web application for turning text or CSV pose data into an animated signing avatar. It combines the browser-based [Pose Animator](https://github.com/yemount/pose-animator) player with a small Python server that generates animation sequences from a local dataset.

> **Privacy:** the `SLclean/` dataset is deliberately excluded from this repository. Add your own CSV dataset locally before using text-to-animation generation.

## What it does

- Plays pose-animation sequences in the browser.
- Generates a sequence from a text query by looking up matching CSV clips in a local dataset.
- Supports speech-to-text input through an optional local Vosk model.
- Lets you load built-in avatars or your own SVG illustration.

## Requirements

- Windows with Python 3.10 or newer
- A modern web browser
- Optional: the [`vosk`](https://pypi.org/project/vosk/) Python package and a local Vosk English model for microphone transcription

## Getting started

1. Clone the project and open its folder.

   ```powershell
   git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
   cd YOUR-REPOSITORY
   ```

2. Create and activate a virtual environment.

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install the optional speech-recognition dependency if you want microphone transcription.

   ```powershell
   pip install vosk
   ```

4. Put your CSV dataset in `SLclean/` (this folder stays only on your computer). Each clip should be available as a CSV file that can be processed by `export_pose_animator_sequence.py`.

5. If using microphone transcription, download and unpack a compatible Vosk model into `vosk-model-small-en-us-0.15/`, or pass its location with `--vosk-model-dir`.

6. Start the server.

   ```powershell
   .\.venv\Scripts\python.exe .\run_pose_animator_server.py --port 8025
   ```

   Or double-click `open_signing_avatar.cmd` after creating `.venv`.

7. Open the address shown in the terminal, normally:

   ```text
   http://127.0.0.1:8025/dataset_player.html
   ```

## Using the app

- Enter text in the player and choose **Generate From Text** to look up matching local CSV clips.
- To play a saved animation, load a JSON sequence in the player.
- Use the controls to select a built-in avatar or load your own SVG illustration.

### Export a sequence manually

```powershell
.\.venv\Scripts\python.exe .\export_pose_animator_sequence.py `
  --data-dir ".\SLclean" `
  --files "SLclean\a.csv" `
  --output ".\pose-animator\resources\data\my_sequence.json"
```

Then load `pose-animator/resources/data/my_sequence.json` in the browser player.

## Project layout

| Path | Purpose |
| --- | --- |
| `run_pose_animator_server.py` | Local web server and text/voice API |
| `export_pose_animator_sequence.py` | Converts pose CSV clips into player JSON |
| `pose-animator/` | Browser player and rendering assets |
| `SLclean/` | Your private local dataset — not published |

## Keeping data private

The `.gitignore` file explicitly excludes `SLclean/`, the downloaded Vosk model, virtual environments, backups, and generated temporary files. Before publishing, use `git status` to confirm that no private files are staged.

## Credits

The browser animation component is based on Pose Animator. See the license and documentation in [`pose-animator/`](pose-animator/).
