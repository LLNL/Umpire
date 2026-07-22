#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
###############################################################################

import hashlib
import json
import os
import subprocess
import sys


def env(name, default=""):
    return os.environ.get(name, default)


def file_hash(path):
    if not os.path.isfile(path):
        return "missing"
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def git_commit(path):
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def sh_quote(value):
    return "'" + value.replace("'", "'\"'\"'") + "'"


def uberenv_config(project_dir):
    with open(os.path.join(project_dir, ".uberenv_config.json")) as fh:
        return json.load(fh)


def cache_target():
    upstream_target = env("UMPIRE_CI_UPSTREAM_TARGET_VALUE", "develop")
    target = env("UMPIRE_CI_CACHE_TARGET_VALUE")
    if target:
        return target
    if env("CI_MERGE_REQUEST_IID_VALUE"):
        return "mr-" + env("CI_MERGE_REQUEST_IID_VALUE")
    if env("CI_COMMIT_BRANCH_VALUE") == env("CI_DEFAULT_BRANCH_VALUE", upstream_target):
        return upstream_target
    if env("CI_COMMIT_REF_SLUG_VALUE"):
        return "ref-" + env("CI_COMMIT_REF_SLUG_VALUE")
    return "manual"


def cache_identity(project_dir):
    config = uberenv_config(project_dir)
    return "\n".join(
        [
            "cache-format=umpire-ci-v1",
            "spec=" + env("SPEC_VALUE"),
            "module-list=" + env("MODULE_LIST_VALUE"),
            "sys-type=" + env("SYS_TYPE_VALUE", "unknown"),
            "machine=" + env("MACHINE_VALUE"),
            "uberenv-config-hash=" + file_hash(os.path.join(project_dir, ".uberenv_config.json")),
            "uberenv-commit=" + git_commit(os.path.join(project_dir, "scripts/uberenv")),
            "radiuss-spack-configs-commit="
            + git_commit(os.path.join(project_dir, "scripts/radiuss-spack-configs")),
            "spack-url=" + config.get("spack_url", ""),
            "spack-branch=" + config.get("spack_branch", ""),
            "spack-commit=" + config.get("spack_commit", ""),
        ]
    )


def prepare():
    project_dir = env("PROJECT_DIR")
    target = cache_target()
    key = hashlib.sha256(cache_identity(project_dir).encode()).hexdigest()
    print("cache_target=" + sh_quote(target))
    print("cache_key=" + sh_quote(key))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] != "prepare":
        sys.exit("usage: umpire_ci_cache.py prepare")
    prepare()
