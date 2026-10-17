"""ASGI API for a persistent Python host; Vercel serves the browser frontend."""
import hashlib
import io
import json
import logging
import os
import threading
import wave
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from urllib.request import urlopen

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from clip_database import database_clips
from export_pose_animator_sequence import build_payload
from run_pose_animator_server import PoseAnimatorHandler

log = logging.getLogger('cloud_backend')
DATA_PATH = Path(os.getenv('DATA_PATH', '/data/SLclean.sqlite3'))
state = {'dataset': 'loading', 'datasetError': None}
work_lock = threading.Lock()


class CloudSpeech(PoseAnimatorHandler):
    """Use the existing WAV decoder with a download-capable cloud model loader."""
    _asr_lock = threading.Lock()
    _asr_state = 'loading'
    _asr_pipeline = None
    _asr_error = None

    @classmethod
    def _load_asr_pipeline(cls):
        if cls._asr_pipeline is None:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline
            processor = Wav2Vec2Processor.from_pretrained(cls.asr_model_id)
            model = Wav2Vec2ForCTC.from_pretrained(cls.asr_model_id)
            cls._asr_pipeline = pipeline('automatic-speech-recognition', model=model,
                tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor, device=-1)
        return cls._asr_pipeline


def prepare_dataset():
    try:
        if not DATA_PATH.exists():
            url = os.getenv('DATASET_DOWNLOAD_URL', '')
            if not url.startswith('https://'):
                raise ValueError('Set an HTTPS DATASET_DOWNLOAD_URL or mount the database file.')
            DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            temporary = DATA_PATH.with_suffix('.downloading')
            digest = hashlib.sha256()
            with urlopen(url, timeout=120) as response, temporary.open('wb') as output:
                while chunk := response.read(4 * 1024 * 1024):
                    digest.update(chunk)
                    output.write(chunk)
            expected = os.getenv('DATASET_SHA256', '').strip().lower()
            if expected and digest.hexdigest() != expected:
                raise ValueError('Database checksum mismatch.')
            # Confirm the downloaded file is our SQLite dataset, not a login/HTML page.
            if not database_clips(temporary):
                raise ValueError('Dataset contains no clips.')
            temporary.replace(DATA_PATH)
        if not database_clips(DATA_PATH):
            raise ValueError('Dataset contains no clips.')
        state.update(dataset='ready', datasetError=None)
    except Exception as exc:
        # Do not publish download URLs, signed query parameters or credentials.
        state.update(dataset='error', datasetError='Database setup failed; check the file, download link and checksum.')
        log.error('Database setup failed (%s)', type(exc).__name__)


def prepare_services():
    prepare_dataset()
    if os.getenv('ENABLE_SPEECH', 'true').lower() == 'true':
        CloudSpeech._preload_asr()
    else:
        CloudSpeech._asr_state = 'disabled'


@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=prepare_services, daemon=True, name='cloud-setup').start()
    yield


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)
origins = [s.strip().rstrip('/') for s in os.getenv('ALLOWED_ORIGINS', '').split(',') if s.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False,
    allow_methods=['GET', 'POST'], allow_headers=['Content-Type'])


@app.exception_handler(HTTPException)
async def api_error(request, exc):
    return JSONResponse({'ok': False, 'error': str(exc.detail)}, status_code=exc.status_code)


@contextmanager
def work_slot():
    if not work_lock.acquire(blocking=False):
        raise HTTPException(429, 'Server is processing another request. Please try again shortly.')
    try:
        yield
    finally:
        work_lock.release()


def require_dataset():
    if state['dataset'] != 'ready':
        raise HTTPException(503, 'Dataset is not ready yet. Please wait or check server configuration.')


async def bounded_body(request, maximum):
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > maximum:
            raise HTTPException(413, 'Request is too large.')
    return bytes(body)


@app.get('/healthz')
def healthz():
    return {'ok': True}


@app.get('/api/health')
def health():
    return {'ok': True, 'dataDirExists': state['dataset'] == 'ready', 'storage': 'sqlite',
        'datasetState': state['dataset'], 'datasetError': state['datasetError'],
        'stt': {'state': CloudSpeech._asr_state, 'loaded': CloudSpeech._asr_state == 'ready',
                'model': CloudSpeech.asr_model_id, 'error': 'Speech model unavailable' if CloudSpeech._asr_state == 'error' else None}}


@app.get('/api/generate_sequence')
def generate(text: str = Query(min_length=1, max_length=160)):
    require_dataset()
    # Public callers can select text only, never arbitrary paths or expensive tuning values.
    if len(text.split()) > 20:
        raise HTTPException(400, 'Use at most 20 words per request.')
    with work_slot():
        try:
            payload = build_payload(data_dir=DATA_PATH, file_tokens=[], text=text, fps=30,
                width=513, height=513, pause_frames=0, max_frames=0, upsample_factor=2,
                gaussian_sigma=1.2, gaussian_radius=2)
            return {'ok': True, 'payload': payload}
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc


@app.post('/api/compare_pose')
async def compare(request: Request):
    require_dataset()
    raw = await bounded_body(request, 12_000_000)
    try:
        body = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(400, 'Invalid JSON.') from exc

    def run():
        from pose_comparison import compare_request
        with work_slot():
            try:
                return compare_request(DATA_PATH, body)
            except (ValueError, KeyError, TypeError) as exc:
                raise HTTPException(400, str(exc)) from exc
    return await run_in_threadpool(run)


@app.post('/api/transcribe_wav')
async def transcribe(request: Request):
    if CloudSpeech._asr_state != 'ready':
        raise HTTPException(503, 'Speech model is not ready.')
    audio = await bounded_body(request, 2_000_000)

    def run():
        with work_slot():
            try:
                with wave.open(io.BytesIO(audio)) as wav:
                    if wav.getframerate() < 8000 or wav.getnframes() / wav.getframerate() > 20:
                        raise ValueError('Use a WAV recording no longer than 20 seconds.')
                # Existing decoder uses no socket/request state; reuse its WAV validation and resampling.
                decoder = object.__new__(CloudSpeech)
                return {'ok': True, 'text': decoder._transcribe_wav_bytes(audio)}
            except (ValueError, EOFError, wave.Error) as exc:
                raise HTTPException(400, 'Invalid mono PCM WAV recording.') from exc
    return await run_in_threadpool(run)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=int(os.getenv('PORT', '8000')), workers=1)
