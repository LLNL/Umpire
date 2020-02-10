#!/bin/bash
##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

env
function or_die () {
    "$@"
    local status=$?
    if [[ $status != 0 ]] ; then
        echo ERROR $status command: $@
        exit $status
    fi
}

or_die mkdir travis-build
or_die mkdir travis-install
cd travis-build
if [[ "$DO_BUILD" == "yes" ]] ; then
    cmake_flags=("-DCMAKE_CXX_COMPILER=$COMPILER")
    if [[ "$DO_INSTALL" == "yes" ]] ; then
        cmake_flags+=("-DCMAKE_INSTALL_PREFIX=../travis-install")
    fi
    or_die cmake "${cmake_flags[@]}" ${CMAKE_EXTRA_FLAGS} ../
    if [[ ${CMAKE_EXTRA_FLAGS} == *COVERAGE* ]] ; then
      or_die make -j 3
    else
      or_die make -j 3 VERBOSE=1
    fi
    if [[ "$DO_TEST" == "yes" ]] ; then
      or_die ctest -T test --output-on-failure -V
    fi
    if [[ "$DO_INSTALL" == "yes" ]] ; then
      or_die make -j 3 install
    fi
    if [[ "$DO_MEMCHECK" == "yes" ]] ; then
      regex="^Memory Leak - [1-9]+"
      while read -r line; do
          if [[ "$line" =~ $regex ]]; then
            echo "Found leaks: $line"
            cat Testing/Temporary/MemoryChecker.*.log
            exit 1
          fi
      done < <(ctest -E replay\|io\|log\|strategy -T memcheck)
      echo "No leaks detected!"
    fi
fi

exit 0
