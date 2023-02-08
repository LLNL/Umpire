##############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
add_subdirectory(tpl)
add_subdirectory(unit)
add_subdirectory(integration)
add_subdirectory(tools)
add_subdirectory(debug)
if (UMPIRE_ENABLE_DEVELOPER_DEFAULTS)
    add_subdirectory(applications)
endif()
add_subdirectory(install)

umpire_add_code_checks(PREFIX tests RECURSIVE TRUE)
