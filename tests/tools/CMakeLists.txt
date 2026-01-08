##############################################################################
# Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

if (UMPIRE_ENABLE_ASAN AND UMPIRE_ENABLE_SANITIZER_TESTS)
    add_subdirectory(sanitizers)
endif ()
