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

from pixalmlx.glb_normals import export_with_corrected_normals
from pixalmlx.server.app import JobOptions, MAX_REQUEST, Settings, WorkerSupervisor, create_app
from pixalmlx.server.store import JobStore, QueueFull
from pixalmlx.server.worker import execute_job, generation_arguments


STEP_FIELDS = ("ss_steps", "lr_steps", "hr_steps", "texture_steps")


def image_bytes():
    output = io.BytesIO()
    Image.new("RGB", (16, 16), "magenta").save(output, format="PNG")
    return output.getvalue()


@pytest.fixture
def service(tmp_path):
    settings = Settings(data_dir=tmp_path, start_worker=False)
    app = create_app(settings)
    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        yield client, app.state.store


def submit(client, options=None):
    return client.post("/api/jobs", files={"image": ("input.png", image_bytes(), "image/png")},
                       data={"options": json.dumps(options or {})})


def test_api_and_dashboard_work_without_tokens_or_login(service):
    client, _ = service
    dashboard = client.get("/")
    assert dashboard.status_code == 200
    assert "login-dialog" not in dashboard.text
    assert 'id="logout"' not in dashboard.text
    for endpoint in ("health", "jobs", "openapi.json"):
        assert client.get("/api/" + endpoint).status_code == 200
    assert client.get("/api/health", headers={"Authorization": "Bearer obsolete"}).status_code == 200
    schema = client.get("/api/openapi.json")
    assert "securitySchemes" not in schema.json()["components"]
    assert "security" not in schema.json()
    assert all(name in schema.text for name in STEP_FIELDS)
    script = client.get("/static/studio.js").text
    assert "Authorization" not in script
    assert "sessionStorage" not in script


def test_upload_persists_normalized_image_and_soft_target(service):
    client, store = service
    response = submit(client)
    assert response.status_code == 202
    job = response.json()
    assert job["status"] == "queued"
    assert job["options"]["target_faces"] == 20000
    assert job["options"]["max_faces"] == 0
    assert job["options"]["hole_fill_backend"] == "cgal-fan"
    assert all(job["options"][name] == 12 for name in STEP_FIELDS)
    assert (store.directory / "jobs" / job["id"] / "input.png").is_file()
    assert JobStore(store.directory).get(job["id"]) == job
    assert client.get(f"/api/jobs/{job['id']}/input").headers["content-type"] == "image/png"
    assert client.get(f"/api/jobs/{job['id']}/model").status_code == 409
    assert client.get(f"/api/jobs/{job['id']}/report").status_code == 409
    assert client.get("/api/jobs").json()["counts"] == {"queued": 1}


@pytest.mark.parametrize("overrides", [
    {},
    {"ss_steps": 5, "lr_steps": 7, "hr_steps": 9, "texture_steps": 11},
    {"hr_steps": 20},
    dict.fromkeys(STEP_FIELDS, 1),
    dict.fromkeys(STEP_FIELDS, 100),
])
def test_sampling_steps_are_persisted_independently_and_forwarded_to_worker(service, overrides):
    client, store = service
    response = submit(client, overrides)
    assert response.status_code == 202
    job = response.json()
    expected = {name: overrides.get(name, 12) for name in STEP_FIELDS}
    assert {name: job["options"][name] for name in STEP_FIELDS} == expected
    assert JobStore(store.directory).get(job["id"])["options"] == job["options"]
    claimed = store.claim()
    arguments = generation_arguments(claimed, store.directory / "jobs" / job["id"])
    for name, value in expected.items():
        option = "--" + name.replace("_", "-")
        assert arguments[arguments.index(option) + 1] == str(value)


@pytest.mark.parametrize("field", STEP_FIELDS)
@pytest.mark.parametrize("value", [-1, 0, 101, 2.5, 12.0, "12", True, None])
def test_sampling_steps_reject_invalid_ranges_and_noninteger_json_types(service, field, value):
    client, store = service
    assert submit(client, {field: value}).status_code == 422
    assert store.counts() == {}


def test_legacy_queued_jobs_default_missing_step_fields_to_twelve(tmp_path):
    options = JobOptions().model_dump()
    for name in STEP_FIELDS:
        del options[name]
    arguments = generation_arguments({"options": options}, tmp_path)
    for name in STEP_FIELDS:
        option = "--" + name.replace("_", "-")
        assert arguments[arguments.index(option) + 1] == "12"


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
    app = create_app(Settings(data_dir=tmp_path, start_worker=False, allow_1024=True, max_pending=1))
    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
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


def test_legacy_tokens_are_ignored_and_security_headers_remain(tmp_path, monkeypatch):
    monkeypatch.setenv("PIXAL3D_API_TOKEN", "obsolete")
    legacy_token = tmp_path / "api-token"
    legacy_token.write_text("obsolete-file-token")
    app = create_app(Settings(data_dir=tmp_path, start_worker=False))
    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        assert "obsolete" not in response.text
        assert response.headers["cache-control"] == "no-store"
        assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
        assert "'wasm-unsafe-eval'" in response.headers["content-security-policy"]
        assert "'unsafe-eval'" not in response.headers["content-security-policy"]
    assert legacy_token.read_text() == "obsolete-file-token"


@pytest.mark.parametrize("origin", ["https://attacker.example", "null", "file:///", "https://127.0.0.1:9999",
                                    "http://user@127.0.0.1:8000", "http://127.0.0.1:8000/path"])
def test_cross_origin_browser_submissions_are_rejected_without_a_token(service, origin):
    client, store = service
    response = client.post("/api/jobs", files={"image": ("input.png", image_bytes(), "image/png")},
                           headers={"Origin": origin})
    assert response.status_code == 403
    assert store.counts() == {}


def test_cross_site_browser_request_without_origin_is_rejected(service):
    client, _ = service
    assert client.get("/api/jobs", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403


@pytest.mark.parametrize("host,origin", [("127.0.0.1:8000", "http://127.0.0.1:8000"),
                                         ("localhost:8000", "http://localhost:8000"),
                                         ("pixal3d.example.ts.net", "https://pixal3d.example.ts.net"),
                                         ("pixal3d.example.ts.net:8443", "https://pixal3d.example.ts.net:8443")])
def test_local_and_tailscale_dashboard_origins_are_allowed(service, host, origin):
    client, _ = service
    assert client.get("/api/health", headers={"Host": host, "Origin": origin}).status_code == 200


@pytest.mark.parametrize("host", ["attacker.example", "pixal3d.example.ts.net.attacker.example"])
def test_untrusted_hosts_are_rejected_to_resist_dns_rebinding(service, host):
    client, _ = service
    assert client.get("/api/health", headers={"Host": host}).status_code == 400


def test_cli_stays_on_loopback_without_creating_a_token(tmp_path, monkeypatch, capsys):
    from pixalmlx.server import __main__ as command
    import pixalmlx.server.app as server_app
    import uvicorn

    static = tmp_path / "static"
    (static / "vendor").mkdir(parents=True)
    (static / "vendor" / "model-viewer.min.js").write_text("")
    monkeypatch.setattr(server_app, "STATIC", static)
    monkeypatch.setenv("PIXAL3D_API_TOKEN", "obsolete")
    monkeypatch.setattr("sys.argv", ["server", "--data-dir", str(tmp_path / "data")])
    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **options: calls.append(options))
    command.main()
    assert calls == [{"host": "127.0.0.1", "port": 8000, "workers": 1}]
    assert not (tmp_path / "data" / "api-token").exists()
    assert "PIXAL3D_API_TOKEN" not in capsys.readouterr().out


def test_supervisor_lock_prevents_two_server_instances(tmp_path):
    class NoThread:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def join(self, timeout):
            pass

    settings = Settings(data_dir=tmp_path / "data", repo_root=tmp_path)
    store = JobStore(settings.data_dir)
    first, second = WorkerSupervisor(settings, store), WorkerSupervisor(settings, store)
    from unittest.mock import patch

    with patch("pixalmlx.server.app.threading.Thread", NoThread):
        first.start()
        try:
            with pytest.raises(RuntimeError, match="already owns"):
                second.start()
        finally:
            first.close()
