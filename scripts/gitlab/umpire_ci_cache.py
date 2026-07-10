#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
###############################################################################

import datetime
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
    target = env("UMPIRE_CI_CACHE_TARGET_VALUE")
    if target:
        return target
    if env("CI_MERGE_REQUEST_IID_VALUE"):
        return "mr-" + env("CI_MERGE_REQUEST_IID_VALUE")
    if env("CI_COMMIT_BRANCH_VALUE") == env("CI_DEFAULT_BRANCH_VALUE", "main"):
        return "main"
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


def metadata(output_path):
    project_dir = env("PROJECT_DIR")
    config = uberenv_config(project_dir)
    data = {
        "cache_format": "umpire-ci-v1",
        "cache_key": env("CACHE_KEY"),
        "cache_target": env("CACHE_TARGET"),
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "spec": env("SPEC_VALUE"),
        "module_list": env("MODULE_LIST_VALUE"),
        "sys_type": env("SYS_TYPE_VALUE", "unknown"),
        "machine": env("MACHINE_VALUE"),
        "spack_url": config.get("spack_url", ""),
        "spack_branch": config.get("spack_branch", ""),
        "spack_config_commit": config.get("spack_commit", ""),
        "spack_commit": git_commit(os.path.join(env("PREFIX"), "spack")),
        "spack_lock_hash": file_hash(os.path.join(env("SPACK_ENV_PATH"), "spack.lock")),
        "uberenv_commit": git_commit(os.path.join(project_dir, "scripts/uberenv")),
        "radiuss_configs_commit": git_commit(os.path.join(project_dir, "scripts/radiuss-spack-configs")),
        "install_tree": env("INSTALL_TREE"),
        "buildcache": env("BUILDCACHE"),
    }
    with open(output_path, "w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("prepare", "metadata"):
        sys.exit("usage: umpire_ci_cache.py prepare|metadata [metadata-file]")
    if sys.argv[1] == "prepare":
        prepare()
    else:
        metadata(sys.argv[2])
