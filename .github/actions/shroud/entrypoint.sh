#!/bin/bash

set -euo pipefail

git config --global --add safe.directory /github/workspace
git config --global --add safe.directory /github/workspace/.radiuss-ci
git config --global --add safe.directory /github/workspace/blt
git config --global --add safe.directory /github/workspace/scripts/radiuss-spack-configs
git config --global --add safe.directory /github/workspace/scripts/uberenv
git config --global --add safe.directory /github/workspace/src/tpl/umpire/camp
git config --global --add safe.directory /github/workspace/src/tpl/umpire/fmt

git submodule update --init --recursive

mkdir build && cd build 
cmake -DCMAKE_CXX_COMPILER=clang++ -DSHROUD_EXECUTABLE=/usr/local/bin/shroud ..
make -j 3 generate_umpire_shroud

# Shroud v0.12.2 can generate a Fortran interface identifier longer than the
# 63-character limit enforced by gfortran, which breaks downstream builds.
# Keep the exported C symbol unchanged (bind(C,name=...)), but shorten the
# internal Fortran procedure name and its call sites.
python3 - <<'PY'
from __future__ import annotations

from pathlib import Path

path = Path("/github/workspace/src/umpire/interface/c_fortran/wrapfumpire.f")
old = "c_resourcemanager_make_allocator_resource_aware_pool_untracked_bufferify"
new = "c_resourcemanager_make_allocator_ra_pool_untracked_bufferify"

if not path.exists():
    raise SystemExit(f"Expected generated file not found: {path}")

text = path.read_text(encoding="utf-8", errors="strict")
if old not in text:
    # If Shroud changes its naming to avoid the long identifier, no-op cleanly.
    raise SystemExit(0)

path.write_text(text.replace(old, new), encoding="utf-8")
PY
