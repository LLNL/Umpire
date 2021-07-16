##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
add_subdirectory(tpl)

add_subdirectory(alloc)
add_subdirectory(resource)
add_subdirectory(detail)
add_subdirectory(allocator)
add_subdirectory(strategy)
add_subdirectory(op)

add_subdirectory(unit)
add_subdirectory(integration)

add_subdirectory(tools)