#!/usr/bin/env bash

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

conf=${CONFIGURATION:-""}
if [[ -z ${conf} ]]
then
    echo "CONFIGURATION is undefined... aborting"
    exit 1
fi

project_dir="$(pwd)"
build_dir="${build_root}/build_${sys_type}_${conf}"
install_dir="${build_root}/install_${sys_type}_${conf}"
option=${1:-""}

# Build
if [[ "${option}" != "--test-only" ]]
then
    # 'conf' = toolchain__tuning
    # 'host_config' = filter__tuning
    #   where 'filter' can represent several toolchains
    #   like <nvcc_10_gcc_X> covers any gcc paired with nvcc10
    # 'toolchain' is a unique set of tools, and 'tuning' allows to have
    # several configurations for this set, like <omptarget>.

    echo "--- Configuration to match :"
    echo "* ${conf}"

    toolchain=${conf/__*/}
    tuning=${conf/${toolchain}/}

    # Find project host_configs matching the configuration
    host_configs="$(ls host-configs/${sys_type}/ | grep "\.cmake$")"
    echo "--- Available host_configs"
    echo "${host_configs}"

    match_count=0
    host_config=""

    # Translate file names into pattern to match the host_config
    echo "--- Patterns"
    for hc in ${host_configs}
    do
        pattern="${hc//X/.*}"
        pattern="${pattern/.cmake/}"
        echo "${pattern}"

        if [[ -n "${tuning}" && ! "${pattern}" =~ .*${tuning}$ ]]
        then
            continue
        fi

        if [[ "${conf}" =~ ^${pattern}$ ]]
        then
            (( ++match_count ))
            host_config="${hc}"
            echo "-> Found Project Conf : ${host_config}"
        fi
    done

    if (( match_count > 1 )) || (( match_count == 0 ))
    then
        echo "ERROR : none or multiple match(s) ..."
        exit 1
    fi

    # If building, then delete everything first
    rm -rf ${build_dir} && mkdir -p ${build_dir} && cd ${build_dir}

    generic_conf="${project_dir}/.radiuss-ci/gitlab/conf/host-configs/${sys_type}/${toolchain}.cmake"
    if [[ ! -f ${generic_conf} ]]
    then
        echo "ERROR: Host-config file ${generic_conf} does not exist" && exit 1
    fi

    project_conf="${project_dir}/host-configs/${sys_type}/${host_config}"
    if [[ ! -f ${project_conf} ]]
    then
        echo "ERROR: Host-config file ${project_conf} does not exist" && exit 1
    fi

    cmake \
      -C ${generic_conf} \
      -C ${project_conf} \
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

    if grep -q "Errors while running CTest" ./tests_output.txt
    then
        echo "ERROR: failure(s) while running CTest" && exit 1
    fi
fi
