#!/usr/bin/env python3
# Copyright Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Builds the sdist and wheel of every published package into dist/.

All packages are released together, so the build fails unless they share one
release version.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path, PurePosixPath

from packaging.utils import parse_sdist_filename, parse_wheel_filename

REPO_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = REPO_ROOT / "dist"
# Files that Python runs on every interpreter start, a known way for a
# tampered wheel to execute code without being imported.
STARTUP_HOOKS = {"sitecustomize.py", "usercustomize.py"}


def is_published(pyproject: Path) -> bool:
    project = tomllib.loads(pyproject.read_text())["project"]
    return "Private :: Do Not Upload" not in project.get("classifiers", [])


def published_package_dirs() -> list[Path]:
    pyprojects = sorted(REPO_ROOT.glob("util/*/pyproject.toml")) + sorted(
        REPO_ROOT.glob("instrumentation/*/pyproject.toml")
    )
    return [path.parent for path in pyprojects if is_published(path)]


def check_wheel_contents(wheel: Path) -> list[str]:
    errors = []
    with zipfile.ZipFile(wheel) as zf:
        for name in zf.namelist():
            path = PurePosixPath(name)
            top = path.parts[0]
            if top != "observeinc" and not top.endswith(".dist-info"):
                errors.append(
                    f"{wheel.name}: unexpected top-level path {name}"
                )
            if path.suffix == ".pth" or path.name in STARTUP_HOOKS:
                errors.append(f"{wheel.name}: startup hook file {name}")
    return errors


def main() -> int:
    package_dirs = published_package_dirs()
    shutil.rmtree(DIST_DIR, ignore_errors=True)

    # Skip anything uploaded in the last 3 days, so a freshly compromised
    # release of a build dependency (e.g. hatchling) can't end up in our build.
    env = {**os.environ, "UV_EXCLUDE_NEWER": "3d"}
    for package_dir in package_dirs:
        subprocess.run(
            ["uv", "build", "--no-sources", "--out-dir", str(DIST_DIR)],
            cwd=package_dir,
            env=env,
            check=True,
        )

    wheels = sorted(DIST_DIR.glob("*.whl"))
    sdists = sorted(DIST_DIR.glob("*.tar.gz"))
    errors = []
    if len(wheels) != len(package_dirs) or len(sdists) != len(package_dirs):
        errors.append(
            f"expected one wheel and one sdist for each of {len(package_dirs)} "
            f"packages, got {len(wheels)} wheels and {len(sdists)} sdists"
        )

    versions = {parse_wheel_filename(w.name)[1] for w in wheels} | {
        parse_sdist_filename(s.name)[1] for s in sdists
    }
    if len(versions) != 1:
        errors.append(
            f"packages must share one version, found {sorted(map(str, versions))}"
        )
    errors += [
        f"{v} is a development version, set the release version first"
        for v in versions
        if v.is_devrelease
    ]

    for wheel in wheels:
        errors += check_wheel_contents(wheel)

    if errors:
        for error in errors:
            print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Built version {versions.pop()}:")
    for artifact in wheels + sdists:
        print(f"  {artifact.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
