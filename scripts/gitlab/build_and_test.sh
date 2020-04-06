#!/bin/bash
##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

set -o errexit
set -o nounset

# Check environment variables
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
    echo "COMPILER is undefined... aborting" && exit 1
fi

project_dir="$(pwd)"
build_dir="${build_root}/build_${sys_type}_${compiler}"
option=${1:-""}

# Build
if [[ "${option}" != "--test-only" ]]
then
    # If building, then delete everything first
    rm -rf ${build_dir} && mkdir -p ${build_dir} && cd ${build_dir}

    conf_suffix="host-configs/${sys_type}/${compiler}.cmake"

    generic_conf="${project_dir}/.radiuss-ci/gitlab/conf/${conf_suffix}"
    if [[ ! -f ${generic_conf} ]]
    then
        echo "ERROR: Host-config file ${generic_conf} does not exist" && exit 1
    fi

    umpire_conf="${project_dir}/${conf_suffix}"
    if [[ ! -f ${umpire_conf} ]]
    then
        echo "ERROR: Host-config file ${umpire_conf} does not exist" && exit 1
    fi

    cmake \
      -C ${generic_conf} \
      -C ${umpire_conf} \
      ${project_dir}
    cmake --build . -j 4
fi

# Test
if [[ "${option}" != "--build-only" ]]
then
    if [[ ! -d ${build_dir} ]]
    then
        echo "ERROR: Build directory not found : ${build_dir}" && exit 1
    fi

    cd ${build_dir}

    ctest -T test 2>&1 | tee tests_output.txt

    no_test_str="No tests were found!!!"
    if [[ "$(tail -n 1 tests_output.txt)" == "${no_test_str}" ]]
    then
        echo "ERROR: No tests were found" && exit 1
    fi

    echo "Copying Testing xml reports for export"
    tree Testing
    cp Testing/*/Test.xml ${project_dir}
fi
