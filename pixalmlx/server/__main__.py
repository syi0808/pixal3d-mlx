import argparse
from pathlib import Path


def main():
    from .app import ROOT, STATIC, Settings, create_app
    import uvicorn

    parser = argparse.ArgumentParser(description="Private Pixal3D API and dashboard; use Tailscale Serve for remote access")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "outputs" / "server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--allow-1024", action="store_true")
    parser.add_argument("--max-pending", type=int, default=20)
    parser.add_argument("--idle-cache-seconds", type=float, default=300)
    arguments = parser.parse_args()
    if not 1 <= arguments.port <= 65535 or arguments.max_pending < 1 or arguments.idle_cache_seconds < 0:
        parser.error("Use a valid port, positive queue limit, and nonnegative idle timeout")
    if not (STATIC / "vendor" / "model-viewer.min.js").is_file():
        parser.error("Build the local GLB viewer first: npm ci --prefix web && npm run build --prefix web")
    arguments.data_dir = arguments.data_dir.expanduser().resolve()
    settings = Settings(data_dir=arguments.data_dir, allow_1024=arguments.allow_1024,
                        max_pending=arguments.max_pending, idle_cache_seconds=arguments.idle_cache_seconds)
    print(f"Dashboard: http://127.0.0.1:{arguments.port}", flush=True)
    print("No API token or login required. Restrict remote access with Tailscale policy; do not use Funnel.", flush=True)
    uvicorn.run(create_app(settings), host="127.0.0.1", port=arguments.port, workers=1)


if __name__ == "__main__":
    main()
