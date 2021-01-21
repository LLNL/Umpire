##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
if (ENABLE_DEVELOPER_BENCHMARKS)
    add_subdirectory(dev_benchmarks)
endif()

if (ENABLE_BENCHMARKS)
    add_subdirectory(user_benchmarks)
endif()

umpire_add_code_checks(PREFIX benchmarks RECURSIVE TRUE)
