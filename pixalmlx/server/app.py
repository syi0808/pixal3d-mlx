from contextlib import asynccontextmanager
from dataclasses import dataclass
import fcntl
import io
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
from typing import Literal
import warnings
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .store import JobStore, QueueFull
from ..cgal_holes import DEFAULT_HOLE_FILL_BACKEND


ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).with_name("static")
MAX_UPLOAD = 16 * 1024 * 1024
MAX_REQUEST = MAX_UPLOAD + 64 * 1024


class JobOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    seed: int = Field(default=42, ge=0, le=2 ** 32 - 4)
    ss_steps: int = Field(default=12, ge=1, le=100)
    lr_steps: int = Field(default=12, ge=1, le=100)
    hr_steps: int = Field(default=12, ge=1, le=100)
    texture_steps: int = Field(default=12, ge=1, le=100)
    resolution: int = 512
    target_faces: int = Field(default=20000, ge=500, le=200000)
    max_faces: int = Field(default=0, ge=0, le=200000)
    texture_size: int = 1024
    remove_background: bool = True
    hole_fill_backend: Literal["cgal-fan", "fan", "cgal"] = DEFAULT_HOLE_FILL_BACKEND

    @field_validator("resolution", "texture_size")
    @classmethod
    def supported_size(cls, value):
        if value not in (512, 1024):
            raise ValueError("Choose 512 or 1024")
        return value

    @field_validator("max_faces")
    @classmethod
    def supported_cap(cls, value):
        if 0 < value < 500:
            raise ValueError("Use 0 for no strict cap, or at least 500 triangles")
        return value


@dataclass
class Settings:
    data_dir: Path
    repo_root: Path = ROOT
    start_worker: bool = True
    allow_1024: bool = False
    max_pending: int = 20
    idle_cache_seconds: float = 300


class WorkerSupervisor:
    def __init__(self, settings, store):
        self.settings, self.store = settings, store
        self.process = None
        self.stopping = threading.Event()
        self.thread = None
        self.lock = None

    def start(self):
        lock_path = self.settings.repo_root / "outputs" / ".pixal3d-server.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = lock_path.open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            self.lock = None
            raise RuntimeError("Another Pixal3D server/worker already owns this checkout") from None
        self.store.interrupt_running("The previous server stopped before this job completed; submit again to retry")
        self.thread = threading.Thread(target=self.supervise, name="pixal3d-worker-supervisor", daemon=True)
        self.thread.start()

    def supervise(self):
        while not self.stopping.is_set():
            with (self.store.directory / "worker.log").open("a", buffering=1) as log:
                try:
                    self.process = subprocess.Popen(
                        [sys.executable, "-u", "-m", "pixalmlx.server.worker", "--data-dir", str(self.store.directory),
                         "--parent-pid", str(os.getpid()), "--idle-cache-seconds", str(self.settings.idle_cache_seconds)],
                        cwd=self.settings.repo_root, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                        start_new_session=True, pass_fds=(self.lock.fileno(),))
                    while self.process.poll() is None:
                        if self.stopping.wait(0.5):
                            self.terminate()
                            break
                    self.store.interrupt_running("The MLX worker stopped unexpectedly; see the worker log and submit again")
                except Exception as error:
                    print(f"Worker supervisor: {error}", file=log, flush=True)
            self.stopping.wait(2)

    def terminate(self):
        process = self.process
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            except ProcessLookupError:
                pass

    def close(self):
        self.stopping.set()
        if self.thread is not None:
            self.thread.join(timeout=10)
        self.terminate()
        self.store.interrupt_running("Server shutdown interrupted this job; submit again to retry")
        if self.lock is not None:
            self.lock.close()

    def alive(self):
        return self.process is not None and self.process.poll() is None


class ServiceMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        if scope["path"].startswith("/api/"):
            origin = headers.get(b"origin")
            same_host = True
            if origin is not None:
                try:
                    source = urlsplit(origin.decode("ascii"))
                    destination = urlsplit("//" + headers.get(b"host", b"").decode("ascii"))
                    default_port = 443 if source.scheme == "https" else 80
                    same_host = (source.scheme in ("http", "https") and source.hostname == destination.hostname
                                 and (source.port or default_port) == (destination.port or default_port)
                                 and not (source.username or source.password or source.path or source.query or source.fragment))
                except (ValueError, UnicodeError):
                    same_host = False
            if not same_host or headers.get(b"sec-fetch-site") == b"cross-site":
                return await JSONResponse({"detail": "Cross-origin browser requests are not allowed"}, status_code=403)(scope, receive, send)
        try:
            length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            return await JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)(scope, receive, send)
        if length > MAX_REQUEST:
            return await JSONResponse({"detail": "Upload exceeds 16 MiB"}, status_code=413)(scope, receive, send)
        consumed = 0

        async def limited_receive():
            nonlocal consumed
            message = await receive()
            consumed += len(message.get("body", b""))
            if consumed > MAX_REQUEST:
                raise StarletteHTTPException(status_code=413, detail="Upload exceeds 16 MiB")
            return message

        async def secure_send(message):
            if message["type"] == "http.response.start":
                message["headers"] = list(message.get("headers", [])) + [
                    (b"cache-control", b"no-store"), (b"x-content-type-options", b"nosniff"),
                    (b"referrer-policy", b"no-referrer"),
                    (b"content-security-policy", b"default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self' 'unsafe-inline'; "
                     b"img-src 'self' blob: data:; connect-src 'self' blob:; worker-src 'self' blob:; "
                     b"object-src 'none'; base-uri 'none'; frame-ancestors 'none'")]
            await send(message)

        await self.app(scope, limited_receive, secure_send)


def create_app(settings):
    directory = Path(settings.data_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(0o700)
    store = JobStore(directory)
    supervisor = WorkerSupervisor(settings, store)

    @asynccontextmanager
    async def lifespan(app):
        if settings.start_worker:
            supervisor.start()
        try:
            yield
        finally:
            if settings.start_worker:
                supervisor.close()

    app = FastAPI(title="Pixal3D Studio", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store, app.state.supervisor = store, supervisor
    upload_lock = threading.Lock()
    app.add_middleware(ServiceMiddleware)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "*.ts.net"], www_redirect=False)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    def find_job(job_id):
        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise HTTPException(404, "Job not found")
        job = store.get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        return job

    @app.get("/")
    def dashboard():
        return FileResponse(STATIC / "index.html", media_type="text/html")

    @app.get("/api/health")
    def health():
        return {"status": "ok", "worker_alive": supervisor.alive(), "counts": store.counts(),
                "allow_1024": settings.allow_1024, "max_pending": settings.max_pending,
                "max_upload_bytes": MAX_UPLOAD, "gpu_concurrency": 1,
                "viewer_ready": (STATIC / "vendor" / "model-viewer.min.js").is_file()}

    @app.get("/api/openapi.json")
    def schema():
        return app.openapi()

    @app.get("/api/jobs")
    def jobs(limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0)):
        return {"jobs": store.list(limit, offset), "counts": store.counts()}

    def prepare_job(image, options):
        try:
            parsed = JobOptions.model_validate_json(options)
        except ValidationError as error:
            raise HTTPException(422, json.loads(error.json(include_url=False))) from None
        if parsed.resolution == 1024 and not settings.allow_1024:
            raise HTTPException(422, "1024 generation requires starting the server with --allow-1024")
        payload = image.file.read(MAX_UPLOAD + 1)
        if len(payload) > MAX_UPLOAD:
            raise HTTPException(413, "Upload exceeds 16 MiB")
        if shutil.disk_usage(directory).free < 2 * 1024 ** 3:
            raise HTTPException(507, "At least 2 GiB of free disk space is required")
        job_id = secrets.token_hex(16)
        job_directory = directory / "jobs" / job_id
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(payload)) as source:
                    if source.format not in ("PNG", "JPEG", "WEBP") or source.width * source.height > 16_000_000:
                        raise ValueError("Use a PNG, JPEG, or WebP image of at most 16 megapixels")
                    if getattr(source, "is_animated", False):
                        raise ValueError("Animated images are not supported")
                    source.load()
                    has_alpha = "A" in source.getbands() or "transparency" in source.info
                    normalized = ImageOps.exif_transpose(source).convert("RGBA" if has_alpha else "RGB")
                    normalized.info.clear()
                    job_directory.mkdir(parents=True)
                    normalized.save(job_directory / "input.png")
            name = Path((image.filename or "image").replace("\\", "/")).name[:120]
            return store.submit(job_id, name, parsed.model_dump(), max_pending=settings.max_pending)
        except QueueFull as error:
            shutil.rmtree(job_directory, ignore_errors=True)
            raise HTTPException(429, str(error)) from None
        except (UnidentifiedImageError, ValueError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
            shutil.rmtree(job_directory, ignore_errors=True)
            raise HTTPException(422, f"Invalid image: {error}") from None

    @app.post("/api/jobs", status_code=202)
    def submit(image: UploadFile = File(...), options: str = Form(
            "{}", description=("JSON object: seed, resolution, target_faces, max_faces, texture_size, remove_background, hole_fill_backend; "
                               "ss_steps, lr_steps, hr_steps, texture_steps (integers 1-100, default 12 each)"))):
        if not upload_lock.acquire(blocking=False):
            raise HTTPException(429, "Another upload is being decoded; retry shortly")
        try:
            return prepare_job(image, options)
        finally:
            upload_lock.release()

    @app.get("/api/jobs/{job_id}")
    def job(job=Depends(find_job)):
        return job

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job=Depends(find_job)):
        if not store.cancel(job["id"]):
            raise HTTPException(409, "Only queued jobs can be cancelled")
        return store.get(job["id"])

    @app.get("/api/jobs/{job_id}/input")
    def input_image(job=Depends(find_job)):
        return FileResponse(directory / "jobs" / job["id"] / "input.png", media_type="image/png")

    @app.get("/api/jobs/{job_id}/model")
    def model(job=Depends(find_job)):
        if job["status"] != "complete":
            raise HTTPException(409, "The corrected model is not ready")
        path = directory / "jobs" / job["id"] / "model.glb"
        if not path.is_file():
            raise HTTPException(404, "Model file is missing")
        return FileResponse(path, media_type="model/gltf-binary", filename=f"pixal3d-{job['id']}.glb")

    @app.get("/api/jobs/{job_id}/report")
    def report(job=Depends(find_job)):
        path = directory / "jobs" / job["id"] / "report.json"
        if job["status"] != "complete" or not path.is_file():
            raise HTTPException(409, "The final report is not ready")
        return FileResponse(path, media_type="application/json", filename=f"pixal3d-{job['id']}.json")

    @app.get("/api/jobs/{job_id}/logs", response_class=PlainTextResponse)
    def logs(job=Depends(find_job)):
        path = directory / "jobs" / job["id"] / "generation.log"
        if not path.is_file():
            return "Waiting for worker.\n"
        with path.open("rb") as source:
            source.seek(max(0, path.stat().st_size - 65536))
            return source.read().decode("utf-8", errors="replace")

    return app
