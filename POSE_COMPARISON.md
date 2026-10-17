# Experimental AI pose comparison

Setup (inside the project directory):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-comparison.txt
.\.venv\Scripts\python.exe setup_comparison.py
```

Restart the Python server, refresh the player, Generate one dataset clip such as
`you` or `high school`, then press **ตรวจด้วย MediaPipe** below Output.
The UI samples up to 12 frames from the actual rendered canvas (white background),
restores playback, and sends those PNGs to the local `/api/compare_pose` endpoint.
Images are not sent to a cloud AI service. Model bundles download from Google's
official MediaPipe storage once; models are excluded from Git.

The server independently reads the original compressed CSV from the configured
SQLite dataset. No smoothing, hand repair, generated pose or skeleton coordinates
are used as the image-side prediction. Source normalized x is multiplied by its
estimated image aspect ratio to avoid treating normalized x/y as square pixels.
Reference frames are nearest timestamps at equal clip progress. This is approximate
alignment, not DTW, and does not evaluate all frames or multi-clip transitions.

Models: MediaPipe Pose Landmarker Full and Hand Landmarker, image mode, CPU,
default 0.5 detection/presence thresholds. Pose landmarks also require visibility
and presence >= 0.5 and coordinates inside the captured image. Hands are assigned
by proximity to detected anatomical pose wrists; ambiguous pairs are omitted.
This heuristic can fail on occluded/crossed arms, so inspect the images and coverage.

Metrics (2D only, no correctness percentage or pass/fail):
- Wrist position distance after shoulder-center translation and shoulder-width scaling.
- Absolute elbow-angle difference, averaged over available arms.
- Finger-joint-angle differences averaged over available paired hands.
- Each mean is averaged over measured frames only; coverage is reported separately.

Missing landmarks are null, never zero error. Reference missing hands are not
filled from earlier frames. A frame selector shows the original captured image
and shoulder-normalized source/AI arm overlay. JSON downloads contain model names,
alignment, coverage, per-frame metrics and arm coordinates, but not captured images.

Limits: Cartoon detection can fail or hallucinate landmarks. Body proportions,
2D projection, source quality and approximate time alignment influence errors.
These are consistency measurements, not proof that a sign has the correct meaning.
The UI also displays heuristic similarity percentages: angle similarity is
`100 * max(0, 1 - error_degrees / 180)`; wrist similarity is `100 / (1 + distance)`.
Each component is averaged across its measured frames. Overall similarity averages
the three equally weighted components per frame, then averages only frames with all
three available. Overall coverage is shown separately. Missing components never
receive a score and cannot produce an overall score. These mappings are not
calibrated AI confidence, recognition accuracy or a pass/fail threshold.
This prototype implements raw source vs image AI (A vs C); it does not yet include
the separate processed-coordinate comparison (A vs B), calibrated thresholds,
DTW, facial grammar, sign recognition or validated motion-path scores.

Official model documentation:
- https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python
- https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python

## Initial local observations (2026-09-26)

Browser tests used the signing avatar, 12 samples and the actual local database.
These values are exploratory measurements, not a model accuracy benchmark.

| Clip | Elbow error / measured frames | Wrist error / measured frames | Finger error / measured frames |
| --- | --- | --- | --- |
| you | 76.21 degrees / 4 of 12 | 1.02 shoulder widths / 4 of 12 | 45.30 degrees / 5 of 12 |
| high school | 11.18 degrees / 12 of 12 | 0.15 shoulder widths / 12 of 12 | 38.91 degrees / 6 of 12 |

A real model run on a blank white image returned no body or hands. Unit tests
check translation/scale invariance, wrong-arm changes, left/right swaps,
missing detections and degenerate angles. The complete Python suite passed 33 tests.
