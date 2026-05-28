#!/usr/bin/env python3
"""Copy apps/breakdown numeric pipeline into vfx_estimator/numeric/bundled/."""

from __future__ import annotations

import argparse
import ast
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLED = ROOT / "vfx_estimator" / "numeric" / "bundled"
SCRIPTS_OUT = BUNDLED / "scripts"
SRC_OUT = BUNDLED / "src"

REQUIRED_SCRIPTS = (
    "generalized_mandays_pipeline.py",
    "tiered_complexity_byzantine.py",
)


def _import_roots(py_path: Path) -> set[str]:
    """Top-level modules imported by a script (for copying src/ml)."""
    tree = ast.parse(py_path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def vendor(source_root: Path) -> None:
    scripts = source_root / "scripts"
    src = source_root / "src"
    if not scripts.is_dir():
        raise FileNotFoundError(f"Missing scripts dir: {scripts}")

    for name in REQUIRED_SCRIPTS:
        if not (scripts / name).is_file():
            raise FileNotFoundError(f"Required script missing: {scripts / name}")

    if BUNDLED.exists():
        shutil.rmtree(BUNDLED)
    SCRIPTS_OUT.mkdir(parents=True)
    SRC_OUT.mkdir(parents=True)

    for name in REQUIRED_SCRIPTS:
        shutil.copy2(scripts / name, SCRIPTS_OUT / name)

    # Copy src tree modules referenced by pipeline (typically ml.*)
    pipeline = SCRIPTS_OUT / "generalized_mandays_pipeline.py"
    roots = _import_roots(pipeline)
    if "tiered_complexity_byzantine" in roots:
        roots.discard("tiered_complexity_byzantine")
    if src.is_dir():
        for mod in sorted(roots):
            mod_path = src / mod
            if mod_path.is_dir():
                shutil.copytree(mod_path, SRC_OUT / mod)
            elif mod_path.with_suffix(".py").is_file():
                shutil.copy2(mod_path.with_suffix(".py"), SRC_OUT / f"{mod}.py")

    # tiered script may import more
    tiered = SCRIPTS_OUT / "tiered_complexity_byzantine.py"
    if tiered.is_file() and src.is_dir():
        for mod in _import_roots(tiered):
            mod_path = src / mod
            if mod_path.is_dir() and not (SRC_OUT / mod).exists():
                shutil.copytree(mod_path, SRC_OUT / mod)

    print(f"Vendored legacy pipeline to {BUNDLED}")
    print(f"  scripts: {list(SCRIPTS_OUT.glob('*.py'))}")
    if SRC_OUT.exists():
        print(f"  src: {[p.relative_to(SRC_OUT) for p in SRC_OUT.rglob('*.py')]}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Vendor apps/breakdown numeric pipeline")
    ap.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Path to apps/breakdown (default: VFX_LEGACY_BREAKDOWN_ROOT from .env)",
    )
    args = ap.parse_args()

    source = args.source
    if source is None:
        sys.path.insert(0, str(ROOT))
        from vfx_estimator.config import get_settings

        get_settings.cache_clear()
        source = get_settings().legacy_breakdown_root

    try:
        vendor(Path(source).resolve())
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print(
            "\nRestore MuseAI-xata/apps/breakdown or pass --source /path/to/apps/breakdown",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
