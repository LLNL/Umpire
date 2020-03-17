#!/bin/bash
##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

set -o errexit
set -o nounset

sys_type=${SYS_TYPE:-""}
if [[ -z ${sys_type} ]]
then
    sys_type=${OSTYPE:-""}
    if [[ -z ${sys_type} ]]
    then
        echo "System type not found (both SYS_TYPE and OSTYPE are undefined)"
        exit 1
    fi
fi

build_root=${BUILD_ROOT:-""}
if [[ -z ${build_root} ]]
then
    build_root=$(pwd)
fi

compiler=${COMPILER:-""}
if [[ -z ${compiler} ]]
then
    echo "COMPILER is undefined... aborting"
    exit 1
fi

project_dir="$(pwd)"
build_dir="${build_root}/build_${sys_type}_${compiler}"
cconf="host-configs/${sys_type}/${compiler}.cmake"

option=${1:-""}

if [[ "${option}" != "--test-only" ]]
then
    # If building, then delete everything first

    rm -rf ${build_dir}
    mkdir -p ${build_dir}
fi

# Assert that build directory exist (mainly for --test-only mode)
if [[ ! -d ${build_dir} ]]
then
    echo "Build directory not found : ${build_dir}"
    exit 1
fi

# Always go to build directory
cd ${build_dir}

# Build
if [[ "${option}" != "--test-only" ]]
then
    cmake \
      -C ${project_dir}/.radiuss-ci/gitlab/conf/${cconf} \
      -C ${project_dir}/${cconf} \
      ${project_dir}
    cmake --build . -j 4
fi

# Test
if [[ "${option}" != "--build-only" ]]
then
    ctest -T test 2>&1 | tee tests_output.txt
    no_test_str="No tests were found\!\!\!"
    [[ "$(tail -n 1 tests_output.txt)" == "${no_test_str}" ]] && exit 1
    tree Testing
    cp Testing/*/Test.xml ${project_dir}
fi
