import argparse
import io
import json
import wave
import webbrowser
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import List
from urllib.parse import parse_qs, urlparse

from export_pose_animator_sequence import build_payload


def _to_int(value: str, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _to_float(value: str, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


class PoseAnimatorHTTPServer(ThreadingHTTPServer):
    """A local server that can be restarted quickly from an IDE."""

    allow_reuse_address = True


def _start_server_on_available_port(host: str, requested_port: int, factory):
    """Use the requested port, or the next free local port when it is busy."""
    last_error = None
    for port in range(int(requested_port), int(requested_port) + 20):
        try:
            return PoseAnimatorHTTPServer((host, port), factory), port
        except OSError as exc:
            last_error = exc
    raise OSError(
        f"Could not start a local server on ports {requested_port}-{requested_port + 19}."
    ) from last_error


class PoseAnimatorHandler(SimpleHTTPRequestHandler):
    data_dir: Path = Path(".")
    default_fps: float = 30.0
    default_width: int = 513
    default_height: int = 513
    default_pause_frames: int = 0
    default_max_frames: int = 0
    default_upsample_factor: int = 2
    default_gaussian_sigma: float = 1.2
    default_gaussian_radius: int = 2
    vosk_model_dir: Path = Path("vosk-model-small-en-us-0.15")
    _vosk_model = None

    def end_headers(self) -> None:
        """Keep the local player in sync with the files being edited."""
        if not self.path.startswith("/api/"):
            self.send_header("Cache-Control", "no-store, max-age=0, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
        super().end_headers()

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _handle_health(self) -> None:
        model_dir = str(self.vosk_model_dir) if self.vosk_model_dir else ""
        model_exists = bool(self.vosk_model_dir and self.vosk_model_dir.exists())
        data_dir_exists = self.data_dir.exists()
        self._send_json(
            HTTPStatus.OK,
            {
                "ok": True,
                "dataDir": str(self.data_dir),
                "dataDirExists": data_dir_exists,
                "stt": {
                    "engine": "vosk",
                    "modelDir": model_dir,
                    "modelExists": model_exists,
                },
            },
        )

    def _get_vosk_model(self):
        if self.__class__._vosk_model is not None:
            return self.__class__._vosk_model

        model_dir = self.vosk_model_dir
        if not model_dir or not model_dir.exists():
            raise RuntimeError(
                f"Vosk model not found: {model_dir}. "
                "Set --vosk-model-dir to a valid vosk model folder."
            )

        try:
            from vosk import Model
        except Exception as exc:
            raise RuntimeError(
                "Python package 'vosk' is not installed. Run: pip install vosk"
            ) from exc

        self.__class__._vosk_model = Model(str(model_dir))
        return self.__class__._vosk_model

    def _transcribe_wav_bytes(self, wav_bytes: bytes) -> str:
        if not wav_bytes:
            raise ValueError("Empty audio payload.")

        try:
            wf = wave.open(io.BytesIO(wav_bytes), "rb")
        except Exception as exc:
            raise ValueError("Invalid WAV data.") from exc

        with wf:
            channels = int(wf.getnchannels())
            sample_width = int(wf.getsampwidth())
            sample_rate = int(wf.getframerate())
            if channels != 1:
                raise ValueError(f"WAV must be mono (1 channel). Got {channels}.")
            if sample_width != 2:
                raise ValueError(
                    f"WAV must be 16-bit PCM (sample width 2). Got {sample_width}."
                )
            if sample_rate < 8000:
                raise ValueError(f"WAV sample rate too low: {sample_rate}.")

            model = self._get_vosk_model()
            from vosk import KaldiRecognizer

            recognizer = KaldiRecognizer(model, float(sample_rate))
            recognizer.SetWords(False)

            segments = []
            while True:
                chunk = wf.readframes(4000)
                if not chunk:
                    break
                if recognizer.AcceptWaveform(chunk):
                    partial_text = str(json.loads(recognizer.Result()).get("text", "")).strip()
                    if partial_text:
                        segments.append(partial_text)

            final_text = str(json.loads(recognizer.FinalResult()).get("text", "")).strip()
            if final_text:
                segments.append(final_text)

            return " ".join(segments).strip()

    def _handle_transcribe_wav(self) -> None:
        content_length = _to_int(str(self.headers.get("Content-Length", "0")), 0)
        if content_length <= 0:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Empty request body."})
            return
        if content_length > 25 * 1024 * 1024:
            self._send_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"ok": False, "error": "Audio payload too large (max 25MB)."},
            )
            return

        wav_bytes = self.rfile.read(content_length)
        try:
            text = self._transcribe_wav_bytes(wav_bytes)
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return
        except RuntimeError as exc:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(exc)})
            return
        except Exception as exc:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(exc)})
            return

        self._send_json(
            HTTPStatus.OK,
            {
                "ok": True,
                "text": text,
            },
        )

    def _handle_generate(self, parsed) -> None:
        query = parse_qs(parsed.query or "", keep_blank_values=False)
        text = str(query.get("text", [""])[0]).strip().lower()

        files: List[str] = []
        for token in query.get("file", []):
            token = str(token).strip()
            if token:
                files.append(token)
        files_arg = str(query.get("files", [""])[0]).strip()
        if files_arg:
            files.extend([item for item in files_arg.split() if item])

        if not text and not files:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {
                    "ok": False,
                    "error": "Provide text or file(s). Example: /api/generate_sequence?text=hello",
                },
            )
            return

        fps = _to_float(str(query.get("fps", [self.default_fps])[0]), self.default_fps)
        width = _to_int(str(query.get("width", [self.default_width])[0]), self.default_width)
        height = _to_int(str(query.get("height", [self.default_height])[0]), self.default_height)
        pause_frames = _to_int(
            str(query.get("pause_frames", [self.default_pause_frames])[0]),
            self.default_pause_frames,
        )
        max_frames = _to_int(
            str(query.get("max_frames", [self.default_max_frames])[0]),
            self.default_max_frames,
        )
        upsample_raw = query.get(
            "upsample",
            query.get("upsample_factor", [self.default_upsample_factor]),
        )[0]
        gaussian_sigma_raw = query.get(
            "gaussian_sigma",
            query.get("gaussianSigma", [self.default_gaussian_sigma]),
        )[0]
        gaussian_radius_raw = query.get(
            "gaussian_radius",
            query.get("gaussianRadius", [self.default_gaussian_radius]),
        )[0]

        upsample_factor = max(
            1,
            _to_int(
                str(upsample_raw),
                self.default_upsample_factor,
            ),
        )
        gaussian_sigma = max(
            0.0,
            _to_float(
                str(gaussian_sigma_raw),
                self.default_gaussian_sigma,
            ),
        )
        gaussian_radius = max(
            0,
            _to_int(
                str(gaussian_radius_raw),
                self.default_gaussian_radius,
            ),
        )

        try:
            payload = build_payload(
                data_dir=self.data_dir,
                file_tokens=files,
                text=text,
                fps=fps,
                width=width,
                height=height,
                pause_frames=pause_frames,
                max_frames=max_frames,
                upsample_factor=upsample_factor,
                gaussian_sigma=gaussian_sigma,
                gaussian_radius=gaussian_radius,
            )
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return
        except Exception as exc:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(exc)})
            return

        self._send_json(HTTPStatus.OK, {"ok": True, "payload": payload})

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self._handle_health()
            return
        if parsed.path == "/api/generate_sequence":
            self._handle_generate(parsed)
            return
        super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/transcribe_wav":
            self._handle_transcribe_wav()
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve pose-animator static files + API for text-to-sequence generation."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8025)
    parser.add_argument(
        "--no-browser",
        dest="open_browser",
        action="store_false",
        help="Start the server without opening the player page.",
    )
    parser.set_defaults(open_browser=True)
    parser.add_argument("--static-dir", default="pose-animator", help="Directory to serve as web root.")
    parser.add_argument("--data-dir", default="SLclean", help="CSV dataset root.")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=513)
    parser.add_argument("--height", type=int, default=513)
    parser.add_argument("--pause-frames", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--upsample-factor", type=int, default=2)
    parser.add_argument("--gaussian-sigma", type=float, default=1.2)
    parser.add_argument("--gaussian-radius", type=int, default=2)
    parser.add_argument(
        "--vosk-model-dir",
        type=str,
        default="vosk-model-small-en-us-0.15",
        help="Path to Vosk model folder (e.g. vosk-model-small-en-us-0.15).",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    static_dir = (base_dir / args.static_dir).resolve()
    data_dir = (base_dir / args.data_dir).resolve()
    vosk_model_dir_arg = Path(str(args.vosk_model_dir))
    if vosk_model_dir_arg.is_absolute():
        vosk_model_dir = vosk_model_dir_arg.resolve()
    else:
        vosk_model_dir = (base_dir / vosk_model_dir_arg).resolve()

    if not static_dir.exists():
        raise SystemExit(f"Static directory not found: {static_dir}")
    class Handler(PoseAnimatorHandler):
        pass

    Handler.data_dir = data_dir
    Handler.default_fps = float(args.fps)
    Handler.default_width = int(args.width)
    Handler.default_height = int(args.height)
    Handler.default_pause_frames = int(args.pause_frames)
    Handler.default_max_frames = int(args.max_frames)
    Handler.default_upsample_factor = max(1, int(args.upsample_factor))
    Handler.default_gaussian_sigma = max(0.0, float(args.gaussian_sigma))
    Handler.default_gaussian_radius = max(0, int(args.gaussian_radius))
    Handler.vosk_model_dir = vosk_model_dir
    Handler._vosk_model = None

    def _factory(*factory_args, **factory_kwargs):
        return Handler(*factory_args, directory=str(static_dir), **factory_kwargs)

    httpd, active_port = _start_server_on_available_port(args.host, int(args.port), _factory)
    player_url = f"http://{args.host}:{active_port}/dataset_player.html?build=signing-avatar-2"
    if active_port != int(args.port):
        print(f"Port {args.port} is busy; using port {active_port} instead.")
    print(f"Serving pose-animator at {player_url}")
    print(f"API: http://{args.host}:{active_port}/api/generate_sequence?text=a")
    print("API: POST /api/transcribe_wav (Content-Type: audio/wav, mono 16-bit PCM)")
    print(f"Vosk model dir: {vosk_model_dir}")
    print(f"Data dir: {data_dir}")
    if not data_dir.exists():
        print("Warning: CSV data directory was not found. The web player is available, but")
        print("         /api/generate_sequence will need --data-dir pointing to your CSV clips.")
    if args.open_browser:
        webbrowser.open(player_url, new=2)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
