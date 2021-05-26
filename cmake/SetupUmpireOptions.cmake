##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
option(UMPIRE_ENABLE_SYCL "Build Umpire with SYCL" Off)
option(UMPIRE_ENABLE_NUMA "Build Umpire with NUMA support" Off)

option(UMPIRE_ENABLE_LOGGING "Build Umpire with Logging enabled" On)
option(UMPIRE_ENABLE_SLIC "Build Umpire with SLIC logging" Off)
option(UMPIRE_ENABLE_BACKTRACE "Build Umpire with allocation backtrace enabled" Off)
option(UMPIRE_ENABLE_BACKTRACE_SYMBOLS "Build Umpire with symbol support" Off)
option(UMPIRE_ENABLE_PEDANTIC_WARNINGS "Enable pedantic compiler warnings" On)
option(UMPIRE_ENABLE_INACCESSIBILITY_TESTS "Test allocator inaccessibility functionality" Off)
option(UMPIRE_ENABLE_TOOLS "Enable Umpire development tools" Off)
option(UMPIRE_ENABLE_DEVICE_CONST "Enable constant memory on GPUs" Off)
option(UMPIRE_ENABLE_PERFORMANCE_TESTS "Enable additional performance tests" Off)
option(UMPIRE_ENABLE_ASAN "Enable use with address sanitizer tools" Off)
option(UMPIRE_ENABLE_SANITIZER_TESTS "Enable address sanitizer tests" Off)
