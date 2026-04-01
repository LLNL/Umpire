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

# Post-process Shroud outputs for toolchain compatibility.
python3 - <<'PY'
from __future__ import annotations

from pathlib import Path


def patch_wrapfumpire() -> None:
    path = Path("/github/workspace/src/umpire/interface/c_fortran/wrapfumpire.f")
    old = "c_resourcemanager_make_allocator_resource_aware_pool_untracked_bufferify"
    new = "c_resourcemanager_make_allocator_ra_pool_untracked_bufferify"

    if not path.exists():
        raise SystemExit(f"Expected generated file not found: {path}")

    text = path.read_text(encoding="utf-8", errors="strict")
    if old not in text:
        # If Shroud changes its naming to avoid the long identifier, no-op cleanly.
        return

    # Shroud v0.12.2 can generate a Fortran interface identifier longer than the
    # 63-character limit enforced by gfortran.
    path.write_text(text.replace(old, new), encoding="utf-8")


def patch_wrapresourcemanager() -> None:
    path = Path("/github/workspace/src/umpire/interface/c_fortran/wrapResourceManager.cpp")
    if not path.exists():
        raise SystemExit(f"Expected generated file not found: {path}")

    text = path.read_text(encoding="utf-8", errors="strict")

    # Newer Umpire Shroud capsules do not contain SWIG-style memory flags.
    text = text.replace("    SHC_rv->cmemflags = SWIG_MEM_RVALUE;\n", "")

    # For the special *_untracked wrappers, Shroud generates the bufferify
    # overloads by calling a (non-existent) C++ ResourceManager method. Patch
    # those calls into the intended makeAllocator<..., false>(...) instantiations.
    text = text.replace(
        "makeAllocator_list_pool_untracked",
        "makeAllocator<umpire::strategy::DynamicPoolList, false>",
    )
    text = text.replace(
        "makeAllocator_quick_pool_untracked",
        "makeAllocator<umpire::strategy::QuickPool, false>",
    )
    text = text.replace(
        "makeAllocator_resource_aware_pool_untracked",
        "makeAllocator<umpire::strategy::ResourceAwarePool, false>",
    )
    text = text.replace(
        "makeAllocator_fixed_pool_untracked",
        "makeAllocator<umpire::strategy::FixedPool, false>",
    )
    text = text.replace(
        "makeAllocator_monotonic_untracked",
        "makeAllocator<umpire::strategy::MonotonicAllocationStrategy, false>",
    )
    text = text.replace(
        "makeAllocator_slot_pool_untracked",
        "makeAllocator<umpire::strategy::SlotPool, false>",
    )
    text = text.replace(
        "makeAllocator_mixed_pool_untracked",
        "makeAllocator<umpire::strategy::MixedPool, false>",
    )

    path.write_text(text, encoding="utf-8")


patch_wrapfumpire()
patch_wrapresourcemanager()
PY
