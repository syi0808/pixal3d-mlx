from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path

from PIL import Image
import pytest
import trimesh

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from trellmlx.glb_normals import export_with_corrected_normals
from trellmlx.server.app import JobOptions, MAX_REQUEST, Settings, WorkerSupervisor, create_app, load_token
from trellmlx.server.store import JobStore, QueueFull
from trellmlx.server.worker import execute_job, generation_arguments


TOKEN = "test-only-pixal3d-api-token-not-production"


def image_bytes():
    output = io.BytesIO()
    Image.new("RGB", (16, 16), "magenta").save(output, format="PNG")
    return output.getvalue()


@pytest.fixture
def service(tmp_path):
    settings = Settings(data_dir=tmp_path, token=TOKEN, start_worker=False)
    app = create_app(settings)
    with TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"}) as client:
        yield client, app.state.store


def submit(client, options=None):
    return client.post("/api/jobs", files={"image": ("input.png", image_bytes(), "image/png")},
                       data={"options": json.dumps(options or {})})


def test_authentication_covers_uploads_models_logs_and_schema(service):
    client, _ = service
    assert client.get("/").status_code == 200
    for endpoint in ("health", "jobs", "openapi.json", "jobs/" + "a" * 32 + "/model", "jobs/" + "a" * 32 + "/logs"):
        assert client.get("/api/" + endpoint, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.post("/api/jobs", headers={"Authorization": ""}).status_code == 401
    schema = client.get("/api/openapi.json")
    assert schema.status_code == 200
    assert schema.json()["components"]["securitySchemes"]["BearerToken"]["scheme"] == "bearer"
    assert schema.json()["security"] == [{"BearerToken": []}]


def test_upload_persists_normalized_image_and_soft_target(service):
    client, store = service
    response = submit(client)
    assert response.status_code == 202
    job = response.json()
    assert job["status"] == "queued"
    assert job["options"]["target_faces"] == 20000
    assert job["options"]["max_faces"] == 0
    assert (store.directory / "jobs" / job["id"] / "input.png").is_file()
    assert JobStore(store.directory).get(job["id"]) == job
    assert client.get(f"/api/jobs/{job['id']}/input").headers["content-type"] == "image/png"
    assert client.get(f"/api/jobs/{job['id']}/model").status_code == 409
    assert client.get(f"/api/jobs/{job['id']}/report").status_code == 409
    assert client.get("/api/jobs").json()["counts"] == {"queued": 1}


def test_palette_transparency_is_preserved_and_metadata_removed(service):
    client, store = service
    image = Image.new("P", (16, 16), 0)
    image.putpalette([255, 0, 255] * 256)
    payload = io.BytesIO()
    image.save(payload, format="PNG", transparency=0, icc_profile=b"test metadata")
    response = client.post("/api/jobs", files={"image": ("transparent.png", payload.getvalue(), "image/png")})
    assert response.status_code == 202
    with Image.open(store.directory / "jobs" / response.json()["id"] / "input.png") as normalized:
        assert normalized.mode == "RGBA"
        assert normalized.getpixel((0, 0))[3] == 0
        assert "icc_profile" not in normalized.info


def test_explicit_1024_opt_in_and_queue_overflow(tmp_path):
    app = create_app(Settings(data_dir=tmp_path, token=TOKEN, start_worker=False, allow_1024=True, max_pending=1))
    with TestClient(app, headers={"Authorization": "Bearer " + TOKEN}) as client:
        assert submit(client, {"resolution": 1024}).status_code == 202
        assert submit(client).status_code == 429
        assert len(list((tmp_path / "jobs").iterdir())) == 1


@pytest.mark.parametrize("options", [{"resolution": 768}, {"resolution": 1024}, {"texture_size": 2048},
                                     {"seed": -1}, {"max_faces": 2}, {"target_faces": 0},
                                     {"shell": "rm -rf /"}, {"seed": "42"}])
def test_unsafe_or_unsupported_generation_options_are_rejected(service, options):
    client, store = service
    assert submit(client, options).status_code == 422
    assert store.counts() == {}


def test_invalid_image_and_request_size_are_rejected(service):
    client, store = service
    response = client.post("/api/jobs", files={"image": ("not-image.png", b"not an image", "image/png")})
    assert response.status_code == 422
    response = client.post("/api/jobs", content=b"a", headers={"content-length": str(MAX_REQUEST + 1)})
    assert response.status_code == 413
    assert store.counts() == {}


def test_chunked_oversize_upload_is_rejected_without_content_length(service):
    client, store = service
    boundary = "pixal3d-boundary"

    def chunks():
        yield f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="x.png"\r\nContent-Type: image/png\r\n\r\n'.encode()
        yield b"x" * (MAX_REQUEST + 1)
        yield f"\r\n--{boundary}--\r\n".encode()

    response = client.post("/api/jobs", content=chunks(), headers={"content-type": f"multipart/form-data; boundary={boundary}"})
    assert response.status_code == 413
    assert store.counts() == {}


def test_queue_claim_is_atomic_and_serial(tmp_path):
    store = JobStore(tmp_path)
    for index in range(5):
        store.submit(f"{index:032x}", "input.png", {})
    with ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(pool.map(lambda _: JobStore(tmp_path).claim(), range(8)))
    claimed = [job for job in claims if job is not None]
    assert len(claimed) == 1
    assert claimed[0]["id"] == "0" * 32
    store.finish(claimed[0]["id"], error="test failure")
    assert store.claim()["id"] == f"{1:032x}"


def test_queue_limit_cancel_and_interrupted_recovery(service):
    client, store = service
    job = submit(client).json()
    with pytest.raises(QueueFull):
        store.submit("b" * 32, "input.png", {}, max_pending=1)
    assert client.post(f"/api/jobs/{job['id']}/cancel").json()["status"] == "cancelled"
    job = submit(client).json()
    store.claim()
    assert client.post(f"/api/jobs/{job['id']}/cancel").status_code == 409
    store.interrupt_running("server stopped")
    recovered = JobStore(store.directory).get(job["id"])
    assert recovered["status"] == "failed"
    assert recovered["stage"] == "interrupted"
    assert recovered["error"] == "server stopped"
    assert store.claim() is None


def test_worker_uses_fixed_paths_and_omits_disabled_cap(tmp_path):
    job = {"options": JobOptions().model_dump()}
    arguments = generation_arguments(job, tmp_path)
    assert "--max-faces" not in arguments
    assert arguments[arguments.index("--output") + 1] == str(tmp_path / "model.glb")
    job["options"]["max_faces"] = 5000
    assert "--max-faces" in generation_arguments(job, tmp_path)


def test_worker_publishes_only_normal_corrected_glb_and_serves_bytes(service):
    client, store = service
    job = submit(client).json()
    job = store.claim()

    def miniature_generator(arguments, runtime):
        output = Path(arguments[arguments.index("--output") + 1])
        mesh = trimesh.creation.icosphere(subdivisions=1)
        export_with_corrected_normals(mesh, output)
        print("fixture export completed")

    execute_job(store, job, None, generate=miniature_generator)
    completed = client.get(f"/api/jobs/{job['id']}").json()
    assert completed["status"] == "complete"
    assert completed["summary"]["normal_correction"] is True
    assert completed["summary"]["triangles"] == 80
    response = client.get(f"/api/jobs/{job['id']}/model")
    assert response.status_code == 200
    assert response.headers["content-type"] == "model/gltf-binary"
    assert hashlib.sha256(response.content).hexdigest() == completed["summary"]["sha256"]
    assert client.get(f"/api/jobs/{job['id']}/report").json()["normals"]["geometry_uv_and_other_attributes_preserved"]
    assert "fixture export completed" in client.get(f"/api/jobs/{job['id']}/logs").text


def test_worker_fails_without_a_matching_normal_audit(service):
    client, store = service
    submit(client)
    job = store.claim()

    def uncorrected_generator(arguments, runtime):
        Path(arguments[arguments.index("--output") + 1]).write_bytes(b"not corrected")

    execute_job(store, job, None, generate=uncorrected_generator)
    assert store.get(job["id"])["status"] == "failed"
    assert client.get(f"/api/jobs/{job['id']}/model").status_code == 409


def test_log_response_is_bounded_and_missing_jobs_are_404(service):
    client, store = service
    job = submit(client).json()
    (store.directory / "jobs" / job["id"] / "generation.log").write_text("x" * 100000)
    assert len(client.get(f"/api/jobs/{job['id']}/logs").content) == 65536
    assert client.get("/api/jobs/not-a-job").status_code == 404
    assert client.get("/api/jobs/" + "c" * 32).status_code == 404
    assert client.get("/static/../../pyproject.toml").status_code == 404


def test_token_is_private_persistent_and_not_exposed_by_health(tmp_path, monkeypatch):
    monkeypatch.delenv("PIXAL3D_API_TOKEN", raising=False)
    token = load_token(tmp_path)
    assert token == load_token(tmp_path)
    assert (tmp_path / "api-token").stat().st_mode & 0o777 == 0o600
    monkeypatch.setenv("PIXAL3D_API_TOKEN", "too-short")
    with pytest.raises(ValueError):
        load_token(tmp_path)
    app = create_app(Settings(data_dir=tmp_path, token=token, start_worker=False))
    with TestClient(app, headers={"Authorization": "Bearer " + token}) as client:
        response = client.get("/api/health")
        assert token not in response.text
        assert response.headers["cache-control"] == "no-store"
        assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
        assert "'wasm-unsafe-eval'" in response.headers["content-security-policy"]
        assert "'unsafe-eval'" not in response.headers["content-security-policy"]


def test_supervisor_lock_prevents_two_server_instances(tmp_path):
    class NoThread:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def join(self, timeout):
            pass

    settings = Settings(data_dir=tmp_path / "data", token=TOKEN, repo_root=tmp_path)
    store = JobStore(settings.data_dir)
    first, second = WorkerSupervisor(settings, store), WorkerSupervisor(settings, store)
    from unittest.mock import patch

    with patch("trellmlx.server.app.threading.Thread", NoThread):
        first.start()
        try:
            with pytest.raises(RuntimeError, match="already owns"):
                second.start()
        finally:
            first.close()
