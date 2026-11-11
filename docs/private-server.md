# Private API, persistent queue, and asset dashboard

This service runs natively on the Apple Silicon Mac. The remote computer only
needs a browser or an HTTP client. It does not load model weights or execute
inference. The dashboard and the API use the same durable SQLite job queue.
No cloud hosting, Redis, Docker, or public ingress is required.

## Setup and start

Run from this checkout with the existing model weights installed:
Node.js/npm is needed only to prepare the self-hosted viewer bundle. The
running API and worker are Python processes. Use an editable install of this
checkout, because the inference entrypoints remain repository scripts.

```bash
uv pip install --python .venv/bin/python -e '.[server,dev]'
npm ci --prefix web
npm run build --prefix web

.venv/bin/python -m pixalmlx.server
```

Open `http://127.0.0.1:8000`. The server listens **only on loopback** and the
dashboard opens directly, with **no login or API token**. The service neither
creates nor reads an `api-token` file, and `PIXAL3D_API_TOKEN` is ignored.
Existing token files are left untouched but are no longer used. The browser
does not store credentials or send an Authorization header.

The model-viewer library is pinned and built into locally served assets. No
CDN is needed for the dashboard or the generated, uncompressed GLBs, whose
PBR textures are embedded. The viewer uses browser-generated neutral lighting.
It is an interactive PBR renderer, not a Blender Cycles reference render.

Options:

```bash
.venv/bin/python -m pixalmlx.server \
  --port 8000 --data-dir outputs/server \
  --max-pending 20 --idle-cache-seconds 300
```

`--allow-1024` explicitly enables 1024 geometry. It is disabled by default
because the 18 GiB machine's prior 1024 run completed with significant memory
pressure and additional swap. It is not a safe-memory guarantee. The service
keeps the existing 512/FP32 generation defaults and 12 sampling steps per
stage. Stage-specific step counts change only when explicitly requested;
there is no automatic step reduction or quantization.

Keep the Mac awake and powered. For attended operation, `caffeinate -i` can
wrap the start command to inhibit idle sleep; it does not make a powered-off
or closed-lid sleeping laptop reachable. For unattended operation, configure
macOS power settings and a user LaunchAgent for the same absolute start
command. Automatic startup is not installed by this change. Ctrl-C stops the
API and its worker; no Blender process is launched by this service.

## Tailscale access

Install Tailscale on both computers, join the same tailnet, and enable HTTPS
for Serve. With the API running locally:

```bash
tailscale serve --bg 8000
tailscale serve status
```

Use the HTTPS `*.ts.net` address printed by Serve for both the dashboard and
API. **Restrict access with tailnet policy:** there is no application-level
authentication. **Do not use Funnel** for this private service. Serve proxies the local
HTTP port; do not expose the whole checkout or outputs directory. This also
avoids the macOS Tailscale variants' restrictions on direct file serving.

The service does not install Tailscale, log in, change tailnet policy, or alter
existing Serve configuration. To remove only this proxy later, use the
matching Serve port's `off` command rather than resetting unrelated services.
See [Serve](https://tailscale.com/docs/features/tailscale-serve) and the
[Serve CLI](https://tailscale.com/docs/reference/tailscale-cli/serve).

## API

All `/api/` endpoints work **without credentials**, including submissions,
health, inputs, logs, reports, and GLBs. Local users and devices permitted by
the Tailscale access policy can view all jobs and submit generation requests.
They share one workspace; this is **not multi-tenant isolation**.

The service accepts localhost/127.0.0.1 and `*.ts.net` Host headers. Other
hosts are rejected to resist DNS rebinding. Browser API requests with a
different Origin host/port or `Sec-Fetch-Site: cross-site` are also rejected;
there is no permissive CORS policy. Ordinary same-host dashboard requests
and command-line clients without an Origin header are supported. These
browser guards are **not authentication** and do not replace Tailscale policy.
Custom non-`ts.net` remote domains are not supported by this configuration.

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Worker readiness, queue counts, allowed resolution |
| POST | `/api/jobs` | Multipart `image` plus JSON-string `options`; returns 202 and job |
| GET | `/api/jobs?limit=50&offset=0` | Paginated job history; limit at most 100 |
| GET | `/api/jobs/{id}` | State, current stage, timestamps, error, final summary |
| POST | `/api/jobs/{id}/cancel` | Cancel a queued job; running jobs return 409 |
| GET | `/api/jobs/{id}/input` | Normalized input PNG |
| GET | `/api/jobs/{id}/logs` | Last 64 KiB of the job log |
| GET | `/api/jobs/{id}/model` | Corrected GLB download; 409 until complete |
| GET | `/api/jobs/{id}/report` | Normal/geometry audit and runtime report |
| GET | `/api/openapi.json` | API schema; no security scheme required |

Example from another computer; no token or Authorization header is needed:

```bash
export PIXAL3D_URL='https://YOUR-MAC.YOUR-TAILNET.ts.net'

curl --fail-with-body "$PIXAL3D_URL/api/health"

curl --fail-with-body "$PIXAL3D_URL/api/jobs" \
  -F 'image=@turnip.png' \
  -F 'options={"seed":42,"resolution":512,"target_faces":20000}'

curl --fail-with-body "$PIXAL3D_URL/api/jobs/JOB_ID"

curl --fail-with-body "$PIXAL3D_URL/api/jobs/JOB_ID/model" \
  -o turnip.glb
```

Do not hold the upload HTTP request open for inference: the 202 response
contains the job ID, and clients poll its state. Closing the browser does not
cancel a queued or running job.

Supported `options` (unknown fields are rejected):

```json
{
  "seed": 42,
  "ss_steps": 12,
  "lr_steps": 12,
  "hr_steps": 12,
  "texture_steps": 12,
  "resolution": 512,
  "target_faces": 20000,
  "max_faces": 0,
  "texture_size": 1024,
  "remove_background": true,
  "hole_fill_backend": "cgal-fan"
}
```

`hole_fill_backend` defaults to `"cgal-fan"`: CGAL refines/fairs suitable
holes, then the reference fan implementation fills rejected holes. Build the
[native helper](cgal-hole-filling.md#build-and-use) before default generation,
or explicitly select `"fan"` to avoid the native dependency. `"cgal"` selects
CGAL without fan fallback. The backend is persisted with each job and included
in runtime metadata; older queued jobs without the field use `"cgal-fan"`.
Restart an existing persistent server/worker after updating the code.

Each step field is optional and independently defaults to **12**. Values must
be integers from **1 through 100**; strings, booleans, fractions, nulls and
out-of-range values return 422 before a job is queued. These fields control
sparse structure, LR shape, HR shape and texture sampling respectively.
Partial overrides preserve the other stages' defaults. For example:

```bash
curl --fail-with-body "$PIXAL3D_URL/api/jobs" \
  -F 'image=@turnip.png' \
  -F 'options={"ss_steps":12,"lr_steps":8,"hr_steps":12,"texture_steps":8}'
```

Step choices are saved in the job options and passed unchanged to inference.
Jobs queued before these fields existed continue to use 12 for missing values.
Equivalent CLI flags are `--ss-steps`, `--lr-steps`, `--hr-steps`, and
`--texture-steps`. Checkpoint resume only reruns texture sampling, so only
`--texture-steps` applies there. Changing step counts changes the sampling
schedule and may change geometry or texture; fewer steps are not an
output-equivalent optimization. The dashboard still submits the defaults
unless a client explicitly supplies overrides through the API.

`target_faces` is a **soft** target. `max_faces: 0` disables the strict cap and
omits the corresponding generation CLI option. To require at most 5,000
triangles, set `target_faces: 5000, max_faces: 5000`. Budgets range from 500 to
200,000; texture size is 512 or 1024; seed is 0 through 4,294,967,292.

Uploads are restricted to PNG/JPEG/WebP, 16 MiB, and 16 megapixels; orientation
is normalized and image metadata is discarded. Only one image decode is
admitted at a time. A streaming request-body limit also covers requests with
no Content-Length. A full queue returns 429; low disk space returns 507.
Client-supplied filenames and paths never control output paths or commands.

## Execution and recovery

- The API process never imports MLX or loads model weights. A supervised,
  separate worker imports MLX lazily when a job is claimed.
- SQLite transactions admit exactly one running job. A checkout-wide OS file
  lock prevents duplicate server instances, including with different data
  directories or accidentally requested multiple Uvicorn workers.
  Do not launch separate standalone GPU generation/benchmark commands while
  serving jobs; those independent CLIs do not acquire the server-instance lock.
- One `InferenceRuntime` is reused across jobs. Its bounded caches expire
  after 300 idle seconds by default. This does not keep every flow model
  resident, and its 10 GB memory guideline is **not a hard process-memory cap**.
- Runtime stage transitions are persisted while inference runs. The dashboard
  polls every 2.5 seconds and shows actual stages, not fabricated percentages.
- Worker crashes fail the active job and trigger a new worker. An independent
  parent watchdog exits the worker if the API disappears, even mid-inference.
  The worker inherits the instance lock until it exits, preventing overlap.
- On server restart, queued jobs resume; interrupted running jobs become
  failed with an explanation. They are **not silently rerun**. Completed jobs
  and artifacts remain accessible. Cancellation is currently queued-only.
- Job records and artifacts are stored under the selected data directory.
  The dashboard shows up to the most recent 100 jobs; API pagination can
  retrieve older records. There is no automatic retention/deletion policy.
  Back up the data directory together; stop the server for a simple SQLite
  file backup, or use SQLite's online backup facilities while it is running.

## Normal-corrected final exports

`generate_pixal3d.py`, `resume_pixal3d.py`, and `reexport_pixal3d.py` now all use
`pixalmlx.glb_normals.export_with_corrected_normals`. The standalone correction
CLI remains available and defaults to no validation cap rather than an
implicit 5K limit.

The shared finalizer exports to a private temporary directory, corrects normals
across UV splits with a 75-degree crease threshold, verifies unit normals and
unchanged per-triangle geometry/UV/other attributes, and preserves embedded
PBR image bytes and material parameters. Only after correction succeeds is
the final GLB atomically published. A sibling `model.normals.json` records
triangle count and preservation checks. No subdivision or texture repainting
is performed, and topology/silhouette defects are not repaired by this step.

The API worker additionally verifies that the downloaded model's SHA-256
matches its normal audit before marking the job complete. Failed or partial
models are never offered through the download or viewer endpoints. Existing
GLB artifacts are not modified in place by this change.

## Validation

```bash
.venv/bin/python -m pytest -q tests/test_server.py tests/test_mesh_normals.py
node --check pixalmlx/server/static/studio.js
```

The tests exercise token-free access, browser-origin/Host guards, durable serial claims, upload limits
including chunked bodies, input validation, queue cancellation/recovery,
checkout locking, normal-corrected publication, failed-export isolation,
bounded logs, exact download hashes, and uncapped/atomic normal finalization.
They use small real GLB fixtures and do not load the full neural models.
Sampling-step tests cover per-stage defaults and partial overrides, strict
integer/range validation, durable job options, legacy queued jobs, independent
SS/LR/HR/texture sampler forwarding, and texture-step forwarding on resume.

### Initial end-to-end verification on this Mac

On 2026-11-11 (Asia/Seoul), a real HTTP request uploaded the attached turnip
image with seed 42, 512 geometry, 1024 textures, a 20,000 soft triangle target,
and no strict cap. A separate persistent worker ran the full neural pipeline
and corrected final normals. This was not a checkpoint replay or mock job.
This initial run predates the removal of application token authentication;
the token-free API/browser verification is described separately below.

- Generation and final validation: **316.450 seconds**, approximately 5m 16s.
- Final model: **20,034 triangles**, 3,325,012 bytes, zero invalid-length
  normals; geometry/UV attributes and embedded PBR maps pass preservation
  checks. Peak MLX allocation: 8,338,815,618 bytes. This is not a new process
  footprint/swap study or a controlled speed comparison.
- The corrected GLB, HTTP download, and browser-button download all hash to
  `5653bd99da2ae0815187519c603cd4360e8554a642b4de80d0ac6998dc57414a`,
  identical to the previously corrected 20K turnip.
- While the worker was generating, paired job-state and health requests
  remained responsive: median 4.9 ms, maximum 13.3 ms across the recorded
  loopback probes. These are small local observations, not remote latency
  or load-test guarantees.
- Browser QA verifies login, image selection, submission, queued cancellation,
  stage polling, restart-preserved history, interactive PBR display, and the
  authenticated GLB download. The bundled viewer requires narrowly scoped
  `wasm-unsafe-eval`; general JavaScript `unsafe-eval` remains disallowed.
- An idle worker was deliberately killed and replaced by its supervisor in
  approximately 2.49 seconds, preserving completed history. An abrupt API
  process exit was then detected by the idle worker's parent watchdog; it
  exited in approximately 0.54 seconds. The instance lock was released and
  port 8765 closed. These are lifecycle checks, not active-generation crash
  timing guarantees. No verification server or worker is left running.
- The full suite reports **401 passed** and the same three pre-existing
  README onboarding failures. A Starlette/httpx deprecation warning is
  non-fatal. Syntax compilation and `git diff --check` also pass.

Verification artifacts are in `outputs/server-validation/`: `api-probe.json`,
`downloaded-model.glb`, `downloaded-report.json`, `dashboard.jpg`,
`full-tests.log`, `lifecycle.json`, and the SQLite job history with per-job logs/reports. The
initial validation server used a test-only token on loopback; it was not a deployed
Tailscale endpoint or an installed auto-start service.

### Token-free access verification

On 2026-11-11 (Asia/Seoul), token authentication and the login/lock UI were
removed at the user's request. The CLI remains bound to `127.0.0.1`, and
network access must be controlled through Tailscale Serve and tailnet policy.

- All eight checked API routes return 200 without an Authorization header:
  health, history, schema, job detail, input image, logs, final report, and GLB.
  The downloaded GLB still matches the original audited SHA-256 above.
- The browser opens directly, automatically connects to the worker, and loads
  the existing PBR GLB without a login prompt or console warnings/errors.
  This check does not rerun neural inference.
- Host/Origin checks accept simulated HTTPS `*.ts.net` requests and reject
  unrelated hosts and cross-site browser submissions. Actual Tailscale
  installation, login, and remote-device connectivity remain unconfigured.
- The 36 focused server tests pass. The full suite reports **415 passed** and
  the same three pre-existing README onboarding failures. JavaScript/Python
  syntax checks and `git diff --check` also pass.
- The verification tab, API, and worker are all closed afterward; ports 8000
  and 8765 have no remaining listeners.

Artifacts: `outputs/server-validation/token-free-api.json`,
`token-free-dashboard.jpg`, `token-free-server.log`, and
`token-free-full-tests.log`.
