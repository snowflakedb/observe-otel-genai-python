#!/usr/bin/env python3
# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Move this fork's packages from upstream's import paths to our own.

Upstream ships the same code under ``opentelemetry.util.genai`` and
``opentelemetry.instrumentation.genai.*``. Keeping those paths would make our
distributions and upstream's overwrite each other's files when installed
together, so every package moves under ``observeinc``:

    opentelemetry.util.genai               -> observeinc.util.genai
    opentelemetry.instrumentation.genai.*  -> observeinc.instrumentation.genai.*
    opentelemetry.test_util_genai          -> observeinc.test_util_genai

The rename is purely mechanical and gives the same result every time it runs,
so sync_upstream.sh applies it to upstream before merging. Both sides of the
merge then carry the same rename, and only real code changes can conflict.

Changelogs are left alone: they describe releases made under the old paths.

Usage: scripts/rename_namespace.py   (run from anywhere inside the repo)
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

NAMESPACE = "observeinc"

# Module paths that move, without the top-level package.
MODULES = ["util.genai", "instrumentation.genai", "test_util_genai"]

# Pinned so the rename produces the same output on every machine.
RUFF = "ruff@0.16.1"
CONFIG = Path(__file__).resolve().parent.parent / "pyproject.toml"


def replacements() -> list[tuple[str, str]]:
    pairs = []
    for module in MODULES:
        for sep in (".", "/"):
            path = module.replace(".", sep)
            pairs.append(
                (f"opentelemetry{sep}{path}", f"{NAMESPACE}{sep}{path}")
            )
    # The wheel target in each package's pyproject.toml.
    pairs.append(('"src/opentelemetry"', f'"src/{NAMESPACE}"'))
    return pairs


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout


def move_packages(root: Path) -> None:
    for src in sorted(root.glob("*/*/src/opentelemetry")):
        for module in MODULES:
            old_dir = src / module.replace(".", "/")
            if not old_dir.is_dir():
                continue
            new_dir = src.parent / NAMESPACE / module.replace(".", "/")
            new_dir.parent.mkdir(parents=True, exist_ok=True)
            git("mv", str(old_dir), str(new_dir))
        # What is left of opentelemetry/ is empty directories and __pycache__.
        if not any(
            p.is_file() and "__pycache__" not in p.parts
            for p in src.rglob("*")
        ):
            shutil.rmtree(src)


def rewrite_references(root: Path) -> list[Path]:
    pairs = replacements()
    changed = []
    for name in git("ls-files", "-z").split("\0"):
        if not name or Path(name).name == "CHANGELOG.md":
            continue
        path = root / name
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue
        new_text = text
        for old, new in pairs:
            new_text = new_text.replace(old, new)
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            changed.append(path)
    return changed


def reformat(paths: list[Path]) -> None:
    # The new name is shorter and sorts before `opentelemetry`, so wrapped lines
    # and import blocks change. Always use this repo's ruff settings: during a
    # sync the tree being renamed is upstream's, whose settings differ.
    py_files = [str(p) for p in paths if p.suffix == ".py"]
    if not py_files:
        return
    ruff = ["uvx", RUFF]
    config = ["--config", str(CONFIG), "--quiet"]
    subprocess.run(
        [*ruff, "check", *config, "--select", "I", "--fix", *py_files],
        check=True,
    )
    subprocess.run([*ruff, "format", *config, *py_files], check=True)


def main() -> int:
    root = Path(git("rev-parse", "--show-toplevel").strip())
    move_packages(root)
    changed = rewrite_references(root)
    reformat(changed)
    print(f"Renamed to {NAMESPACE}; rewrote {len(changed)} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
