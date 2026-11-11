"""Package naming and distribution contracts."""

import ast
import importlib.util
from pathlib import Path
import tomllib

import pixalmlx


ROOT = Path(__file__).resolve().parents[1]


def test_distribution_packages_the_canonical_namespace():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())

    assert config["project"]["name"] == "pixal3d-mlx"
    assert config["project"]["urls"]["Repository"] == "https://github.com/syi0808/pixal3d-mlx"
    assert config["tool"]["setuptools"]["packages"]["find"]["include"] == ["pixalmlx*"]
    assert "pixalmlx.server" in config["tool"]["setuptools"]["package-data"]
    assert Path(pixalmlx.__file__).resolve().parent == ROOT / "pixalmlx"


def test_module_entrypoints_resolve_inside_the_canonical_package():
    for module_name in ("batch_inference", "checkpoint", "convert", "server.__main__", "server.worker"):
        module = importlib.util.find_spec(f"pixalmlx.{module_name}")

        assert module is not None
        assert Path(module.origin).is_relative_to(ROOT / "pixalmlx")


def test_first_party_python_does_not_import_the_old_namespace():
    sources = list(ROOT.glob("*.py"))
    for directory in ("pixalmlx", "scripts", "tests"):
        sources.extend((ROOT / directory).rglob("*.py"))

    for source in sources:
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            assert all(module.split(".")[0] != "trellmlx" for module in modules), source


def test_viewer_build_targets_the_canonical_package():
    script = (ROOT / "web" / "vendor.mjs").read_text()

    assert "../pixalmlx/server/static/vendor/" in script
