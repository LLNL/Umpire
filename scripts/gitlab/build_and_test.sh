#!/usr/bin/env bash

##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################


set -o errexit
set -o nounset

option=${1:-""}
hostname="$(hostname)"
project_dir="$(pwd)"

build_root=${BUILD_ROOT:-""}
hostconfig=${HOST_CONFIG:-""}
spec=${SPEC:-""}

sys_type=${SYS_TYPE:-""}
py_env_path=${PYTHON_ENVIRONMENT_PATH:-""}

# Dependencies
date
if [[ "${option}" != "--build-only" && "${option}" != "--test-only" ]]
then
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Building Dependencies"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    if [[ -z ${spec} ]]
    then
        echo "SPEC is undefined, aborting..."
        exit 1
    fi

    prefix_opt=""

    if [[ -d /dev/shm ]]
    then
        prefix="/dev/shm/${hostname}/${spec// /_}"
        mkdir -p ${prefix}
        prefix_opt="--prefix=${prefix}"
    fi

    python scripts/uberenv/uberenv.py --spec="${spec}"

fi
date

# Host config file
if [[ -z ${hostconfig} ]]
then
    # If no host config file was provided, we assume it was generated.
    # This means we are looking of a unique one in project dir.
    hostconfigs=( $( ls "${project_dir}/"hc-*.cmake ) )
    if [[ ${#hostconfigs[@]} == 1 ]]
    then
        hostconfig_path=${hostconfigs[0]}
        echo "Found host config file: ${hostconfig_path}"
    elif [[ ${#hostconfigs[@]} == 0 ]]
    then
        echo "No result for: ${project_dir}/hc-*.cmake"
        echo "Spack generated host-config not found."
        exit 1
    else
        echo "More than one result for: ${project_dir}/hc-*.cmake"
        echo "${hostconfigs[@]}"
        echo "Please specify one with HOST_CONFIG variable"
        exit 1
    fi
else
    # Using provided host-config file.
    hostconfig_path="${project_dir}/host-configs/${hostconfig}"
fi

# Build Directory
if [[ -z ${build_root} ]]
then
    build_root=$(pwd)
fi

build_dir="${build_root}/build_${hostconfig//.cmake/}"

cmake_exe=`grep 'CMake executable' ${hostconfig_path} | cut -d ':' -f 2 | xargs`

# Build
if [[ "${option}" != "--deps-only" && "${option}" != "--test-only" ]]
then
    date
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~ Host-config: ${hostconfig_path}"
    echo "~ Build Dir:   ${build_dir}"
    echo "~ Project Dir: ${project_dir}"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo ""
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~ ENV ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Building Umpire"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    # If building, then delete everything first
    rm -rf ${build_dir} 2>/dev/null
    mkdir -p ${build_dir} && cd ${build_dir}

    date
    $cmake_exe \
      -C ${hostconfig_path} \
      ${project_dir}
    if ! $cmake_exe --build . -j; then
      echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
      echo "Compilation failed, running make VERBOSE=1"
      echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
      $cmake_exe --build . --verbose -j 1
    fi
    date
fi

# Test
if [[ "${option}" != "--build-only" ]] && grep -q -i "ENABLE_TESTS.*ON" ${hostconfig_path}
then
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Testing Umpire"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    if [[ ! -d ${build_dir} ]]
    then
        echo "ERROR: Build directory not found : ${build_dir}" && exit 1
    fi

    cd ${build_dir}

    date
    ctest --output-on-failure -T test 2>&1 | tee tests_output.txt
    date

    # If Developer benchmarks enabled, run the no-op benchmark and show output
    if [[ "${option}" != "--build-only" ]] && grep -q -i "ENABLE_DEVELOPER_BENCHMARKS.*ON" ${hostconfig_path}
    then
        date
        ctest -C Benchmark
        date
    fi

    no_test_str="No tests were found!!!"
    if [[ "$(tail -n 1 tests_output.txt)" == "${no_test_str}" ]]
    then
        echo "ERROR: No tests were found" && exit 1
    fi

    echo "Copying Testing xml reports for export"
    tree Testing
    cp Testing/*/Test.xml ${project_dir}

    # Convert CTest xml to JUnit (on toss3 only)
    if [[ ${sys_type} == *toss_3* ]]; then
        if [[ -n ${py_env_path} ]]; then
            . ${py_env_path}/bin/activate

            python3 ${project_dir}/scripts/gitlab/convert_to_junit.py \
            ${project_dir}/Test.xml \
            ${project_dir}/scripts/gitlab/junit.xslt > ${project_dir}/junit.xml
        else
            echo "ERROR: needs python env with lxml, please set PYTHON_ENVIRONMENT_PATH"
        fi
    fi

    if grep -q "Errors while running CTest" ./tests_output.txt
    then
        echo "ERROR: failure(s) while running CTest" && exit 1
    fi
fi
