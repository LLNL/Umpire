#!/usr/bin/env bash

# Initialize modules for users not using bash as a default shell
if test -e /usr/share/lmod/lmod/init/bash
then
  . /usr/share/lmod/lmod/init/bash
fi

###############################################################################
# Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
###############################################################################

set -o errexit
set -o nounset

option=${1:-""}
hostname="$(hostname)"
truehostname=${hostname//[0-9]/}
project_dir="$(pwd)"

hostconfig=${HOST_CONFIG:-""}
spec=${SPEC:-""}
module_list=${MODULE_LIST:-""}
job_unique_id=${CI_JOB_ID:-""}
use_dev_shm=${USE_DEV_SHM:-true}
spack_debug=${SPACK_DEBUG:-false}
debug_mode=${DEBUG_MODE:-false}

if [[ ${debug_mode} == true ]]
then
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Debug mode:"
    echo "~~~~~ - Spack debug mode."
    echo "~~~~~ - Deactivated shared memory."
    echo "~~~~~ - Do not push to buildcache."
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    use_dev_shm=false
    spack_debug=true
fi

# REGISTRY_TOKEN allows you to provide your own personal access token to the CI
# registry. Be sure to set the token with at least read access to the registry.
registry_token=${REGISTRY_TOKEN:-""}
ci_registry_user=${CI_REGISTRY_USER:-"${USER}"}
ci_registry_image=${CI_REGISTRY_IMAGE:-"czregistry.llnl.gov:5050/radiuss/umpire"}
ci_registry_token=${CI_JOB_TOKEN:-"${registry_token}"}

if [[ -n ${module_list} ]]
then
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Modules to load: ${module_list}"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    module load ${module_list}
fi

prefix=""

if [[ -d /dev/shm && ${use_dev_shm} == true ]]
then
    prefix="/dev/shm/${hostname}"
    if [[ -z ${job_unique_id} ]]; then
      job_unique_id=manual_job_$(date +%s)
      while [[ -d ${prefix}-${job_unique_id} ]] ; do
          sleep 1
          job_unique_id=manual_job_$(date +%s)
      done
    fi

    prefix="${prefix}-${job_unique_id}"
    mkdir -p ${prefix}
else
    # We set the prefix in the parent directory so that spack dependencies are not installed inside the source tree.
    prefix="$(pwd)/../spack-and-build-root"
    mkdir -p ${prefix}
fi

spack_cmd="${prefix}/spack/bin/spack"
spack_env_path="${prefix}/spack_env"
uberenv_cmd="./scripts/uberenv/uberenv.py"
if [[ ${spack_debug} == true ]]
then
    spack_cmd="${spack_cmd} --debug --stacktrace"
    uberenv_cmd="${uberenv_cmd} --spack-debug"
fi

# Dependencies
date
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~ Build and test started"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
if [[ "${option}" != "--build-only" && "${option}" != "--test-only" ]]
then
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Building dependencies"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    if [[ -z ${spec} ]]
    then
        echo "SPEC is undefined, aborting..."
        exit 1
    fi

    prefix_opt="--prefix=${prefix}"

    # We force Spack to put all generated files (cache and configuration of
    # all sorts) in a unique location so that there can be no collision
    # with existing or concurrent Spack.
    spack_user_cache="${prefix}/spack-user-cache"
    export SPACK_DISABLE_LOCAL_CONFIG=""
    export SPACK_USER_CACHE_PATH="${spack_user_cache}"
    mkdir -p ${spack_user_cache}

    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Spack setup and environment "
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    ${uberenv_cmd} --setup-and-env-only --spec="${spec}" ${prefix_opt}

    if [[ -n ${ci_registry_token} ]]
    then
        echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        echo "~~~~~ GitLab registry as Spack Build Cache "
        echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        ${spack_cmd} -D ${spack_env_path} mirror add --unsigned --oci-username ${ci_registry_user} --oci-password ${ci_registry_token} gitlab_ci oci://${ci_registry_image}
    fi

    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Spack build of dependencies "
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    ${uberenv_cmd} --skip-setup-and-env --spec="${spec}" ${prefix_opt}

    if [[ -n ${ci_registry_token} && ${debug_mode} == false ]]
    then
        echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        echo "~~~~~ Push dependencies to Build Cache "
        echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        ${spack_cmd} -D ${spack_env_path} buildcache push --only dependencies gitlab_ci
    fi

fi
  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
  echo "~~~~~ Dependencies built"
  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
date

# Host config file
if [[ -z ${hostconfig} ]]
then
    # If no host config file was provided, we assume it was generated.
    # This means we are looking of a unique one in project dir.
    hostconfigs=( $( ls "${project_dir}/"*.cmake ) )
    if [[ ${#hostconfigs[@]} == 1 ]]
    then
        hostconfig_path=${hostconfigs[0]}
        echo "Found host config file: ${hostconfig_path}"
    elif [[ ${#hostconfigs[@]} == 0 ]]
    then
        echo "No result for: ${project_dir}/*.cmake"
        echo "Spack generated host-config not found."
        exit 1
    else
        echo "More than one result for: ${project_dir}/*.cmake"
        echo "${hostconfigs[@]}"
        echo "Please specify one with HOST_CONFIG variable"
        exit 1
    fi
else
    # Using provided host-config file.
    hostconfig_path="${project_dir}/host-configs/${hostconfig}"
fi

hostconfig=$(basename ${hostconfig_path})

# Build Directory
# When using /dev/shm, we use prefix for both spack builds and source build, unless BUILD_ROOT was defined
build_root=${BUILD_ROOT:-"${prefix}"}

build_dir="${build_root}/build_${hostconfig//.cmake/}"
install_dir="${build_root}/install_${hostconfig//.cmake/}"

cmake_exe=`grep 'CMake executable' ${hostconfig_path} | cut -d ':' -f 2 | xargs`

# Build
if [[ "${option}" != "--deps-only" && "${option}" != "--test-only" ]]
then
    date
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Host-config: ${hostconfig_path}"
    echo "~~~~~ Build Dir:   ${build_dir}"
    echo "~~~~~ Project Dir: ${project_dir}"
    echo "~~~~~ Install Dir: ${install_dir}"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo ""
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Building Umpire"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    # If building, then delete everything first
    rm -rf ${build_dir} 2>/dev/null
    mkdir -p ${build_dir} && cd ${build_dir}

    # We set the MPI tests command to allow overlapping.
    # Shared allocation: Allows build_and_test.sh to run within a sub-allocation (see CI config).
    # Use /dev/shm: Prevent MPI tests from running on a node where the build dir doesn't exist.
    cmake_options=""
    if [[ "${truehostname}" == "ruby" || "${truehostname}" == "poodle" ]]
    then
        cmake_options="-DBLT_MPI_COMMAND_APPEND:STRING=--overlap"
    fi

    date
    if [[ "${truehostname}" == "corona" || "${truehostname}" == "tioga" ]]
    then
        module unload rocm
    fi
    $cmake_exe \
      -C ${hostconfig_path} \
      ${cmake_options} \
      -DCMAKE_INSTALL_PREFIX=${install_dir} \
      ${project_dir}
    if ! $cmake_exe --build . -j
    then
        echo "[Error]: compilation failed, building with verbose output..."
        echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        echo "~~~~~ Running make VERBOSE=1"
        echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        $cmake_exe --build . --verbose -j 1
    else
        $cmake_exe --install .
    fi
    date

    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Umpire Built"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
fi

# Test
if [[ "${option}" != "--build-only" ]] && grep -q -i "ENABLE_TESTS.*ON" ${hostconfig_path}
then
    date
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Testing Umpire"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    if [[ ! -d ${build_dir} ]]
    then
        echo "[Error]: Build directory not found : ${build_dir}" && exit 1
    fi

    cd ${build_dir}

    date
    ctest --output-on-failure --no-compress-output -T test -VV 2>&1 | tee tests_output.txt
    date

    # If Developer benchmarks enabled, run the no-op benchmark and show output
    if [[ "${option}" != "--build-only" ]] && grep -q -i "UMPIRE_ENABLE_DEVELOPER_BENCHMARKS.*ON" ${hostconfig_path}
    then
        date
        ctest --verbose -C Benchmark -R no-op_stress_test
        date
    fi

    no_test_str="No tests were found!!!"
    if [[ "$(tail -n 1 tests_output.txt)" == "${no_test_str}" ]]
    then
        echo "[Error]: No tests were found" && exit 1
    fi

    echo "Copying Testing xml reports for export"
    tree Testing
    xsltproc -o junit.xml ${project_dir}/blt/tests/ctest-to-junit.xsl Testing/*/Test.xml
    mv junit.xml ${project_dir}/junit.xml

    if grep -q "Errors while running CTest" ./tests_output.txt
    then
        echo "[Error]: failure(s) while running CTest" && exit 1
    fi

    if grep -q -i "ENABLE_HIP.*ON" ${hostconfig_path}
    then
        echo "[Warning]: not testing install with HIP"
    else
        if [[ ! -d ${install_dir} ]]
        then
            echo "[Error]: install directory not found : ${install_dir}" && exit 1
        fi

        cd ${install_dir}/examples/umpire/using-with-cmake
        mkdir build && cd build
        if ! $cmake_exe -C ../host-config.cmake ..; then
        echo "[Error]: running $cmake_exe for using-with-cmake test" && exit 1
        fi

        if ! make; then
        echo "[Error]: running make for using-with-cmake test" && exit 1
        fi
    fi

    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Umpire Tests Complete"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    date
fi

#echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
#echo "~~~~~ CLEAN UP"
#echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
#make clean

cd ${project_dir}

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~ Build and test completed"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~ To reproduce this build on ${truehostname}:"
echo ""
echo " git checkout `git rev-parse HEAD` && git submodule update --init --recursive"
echo ""
echo "SPACK_DISABLE_LOCAL_CONFIG=\"\" SPACK_USER_CACHE_PATH=\"ci_spack_cache\" python3 scripts/uberenv/uberenv.py --prefix=/tmp/\$(whoami)/uberenv --spec=\"${spec}\""
echo ""
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
date
