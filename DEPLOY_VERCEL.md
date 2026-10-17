# Deploy Sign Language Pose Animator

The Vercel project serves the frontend. A separate persistent Python service runs
generation, speech and MediaPipe comparison. The private 2.7 GB SQLite file is not
included in Git, Vercel uploads or Docker images.

## 1. Create the Vercel frontend

Import `tinnapats/signing-avatar-pose-animator` into Vercel, select branch
`tinnapat`, and keep **Root Directory = repository root**. `vercel.json` sets:

- Framework: Other
- Install: `npm ci --prefix frontend-tooling`
- Build: `npm run build --prefix frontend-tooling`
- Output: `pose-animator-dist`
- `/` opens `dataset_player.html`.

The frontend can deploy first. Generate, microphone and comparison become usable
only after step 3. Without the API connection the page displays a connection notice.
No Python Functions or private models are uploaded to Vercel.

## 2. Create a Python backend (example: Render Docker Web Service)

Create a Docker Web Service from the same repo and branch, using `Dockerfile` at
repository root. Set health-check path `/healthz`. Attach persistent disk mounted
at `/data`; allocate at least 10 GB for the dataset, temporary download and model
cache. Persistent disks may require a paid host plan: review the provider's quote
before creating the service. No paid service is created by these files.

Speech is a large model. Budget several GB of RAM and benchmark the selected
instance; this setup is not validated on a small free instance. Use
`ENABLE_SPEECH=false` to deploy text generation and comparison without preloading
the speech model. Docker uses CPU PyTorch and Uvicorn with one process; a lock
allows one expensive request at a time, returning 429 while busy.

Set these backend environment variables:

| Variable | Value |
| --- | --- |
| `DATA_PATH` | `/data/SLclean.sqlite3` |
| `ALLOWED_ORIGINS` | Exact frontend origin, e.g. `https://your-project.vercel.app` |
| `DATASET_DOWNLOAD_URL` | Private HTTPS link that returns the raw SQLite file |
| `DATASET_SHA256` | Recommended: SHA-256 of the original database |
| `ENABLE_SPEECH` | `true` or `false` |
| `HF_HOME` | `/data/huggingface` (already set in Dockerfile) |

Separate multiple permitted origins with commas. Add preview URLs explicitly if
you want previews to call the backend. CORS is a browser-origin restriction,
not user authentication; consider rate limiting/access control before wide public use.

Compute the database checksum locally:

```powershell
Get-FileHash -LiteralPath C:\pro1end\SLclean.sqlite3 -Algorithm SHA256
```

Place the database in private storage and generate a direct download URL. Ordinary
Google Drive sharing pages return HTML and cannot be used as raw download links.
Keep the URL in backend environment variables only, never in `VITE_*` or Git.
Alternatively, copy the database to the mounted disk using your host's file transfer
tools, and omit the download URL. The backend checks the format before use and
does not re-download when the database already exists. Replacing a dataset is a
separate explicit maintenance action.

The first start downloads the database (if needed) and speech model. `/healthz`
checks the running process; `/api/health` separately reports `datasetState` and
speech readiness. Generation returns 503 until the database is ready. Check host
logs if `datasetState=error`.

## 3. Connect Vercel to the backend

In Vercel > Settings > Environment Variables, add:

```text
VITE_API_BASE_URL=https://your-backend.onrender.com
```

Set it for Production (and Preview if appropriate), then redeploy. This URL is
compiled into the frontend and is public. Use HTTPS so the microphone can work
on the Vercel HTTPS page. The browser calls the backend directly, avoiding putting
large frame/image responses through Vercel Functions. Local development uses
same-origin `/api/...` when the variable is unset.

## 4. Acceptance checks

1. Backend `/api/health`: `dataDirExists=true`, `datasetState=ready`.
2. Open frontend `/`, Generate `I love you` and `you go high school!`.
3. Check Output, playback, single-word preview, Reset, speed and Loop.
4. Generate `high school`, then run MediaPipe comparison.
5. When speech reports ready, allow the mic and test two successive recordings.
6. Test on a second device; it must not depend on `127.0.0.1` or local file paths.

Local preparation includes API/CORS tests and a frontend build. It does not prove
cloud deployment, model performance or host capacity. Docker image execution and
the real public URL must still pass the above checks on the chosen host.
