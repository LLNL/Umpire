##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

if (UMPIRE_HAS_ASAN)
    add_subdirectory(sanitizers)
endif ()
