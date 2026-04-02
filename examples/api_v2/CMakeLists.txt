##############################################################################
# Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set(api_v2_example_targets
  api_v2_basic_host_allocation
  api_v2_stl_containers
  api_v2_cross_platform_copy_example
  api_v2_fixed_pool_example
  api_v2_thread_safe)

blt_add_executable(
  NAME api_v2_basic_host_allocation
  SOURCES basic_host_allocation.cpp
  DEPENDS_ON ${example_depends})

blt_add_executable(
  NAME api_v2_stl_containers
  SOURCES stl_containers.cpp
  DEPENDS_ON ${example_depends})

blt_add_executable(
  NAME api_v2_cross_platform_copy_example
  SOURCES cross_platform_copy.cpp
  DEPENDS_ON ${example_depends})

blt_add_executable(
  NAME api_v2_fixed_pool_example
  SOURCES fixed_pool.cpp
  DEPENDS_ON ${example_depends})

blt_add_executable(
  NAME api_v2_thread_safe
  SOURCES thread_safe.cpp
  DEPENDS_ON ${example_depends})

foreach(api_v2_example_target IN LISTS api_v2_example_targets)
  target_include_directories(
    ${api_v2_example_target}
    BEFORE PRIVATE
    ${PROJECT_SOURCE_DIR}/include)
endforeach()

# On case-insensitive filesystems, the v1 header `Allocator.hpp` can be
# shadowed by the API v2 public header `allocator.hpp` unless the legacy source
# include directories are searched first for operation examples.
target_include_directories(
  api_v2_cross_platform_copy_example
  BEFORE PRIVATE
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/src/umpire
  ${PROJECT_SOURCE_DIR}/include)

install(TARGETS ${api_v2_example_targets} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
