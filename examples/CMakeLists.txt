##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set (example_depends umpire)

if (ENABLE_HIP)
  set (example_depends
      ${example_depends}
      hip_runtime)
endif()

if (ENABLE_OPENMP_TARGET)
  set (example_depends
      ${example_depends}
      openmp)
endif()

blt_add_executable(
  NAME aligned_allocator
  SOURCES aligned_allocator.cpp
  DEPENDS_ON ${example_depends})

blt_add_executable(
  NAME alloc_access
  SOURCES allocator_access.cpp
  DEPENDS_ON ${example_depends})
list(APPEND umpire_examples alloc_access)

blt_add_executable(
  NAME malloc
  SOURCES malloc.cxx
  DEPENDS_ON ${example_depends})
list(APPEND umpire_examples malloc)

if (ENABLE_HIP)
blt_add_executable(
    NAME multi_device
    SOURCES multi_device.cpp
    DEPENDS_ON umpire hip)
list(APPEND umpire_examples multi_device)
endif()

blt_add_executable(
  NAME get_parent 
  SOURCES get_parent.cpp
  DEPENDS_ON ${example_depends})
list(APPEND umpire_examples get_parent)

blt_add_executable(
  NAME strategy_example
  SOURCES strategy_example.cpp
  DEPENDS_ON ${example_depends})
list(APPEND umpire_examples strategy_example)

blt_add_executable(
  NAME allocator
  SOURCES allocator.cxx
  DEPENDS_ON ${example_depends})
list(APPEND umpire_examples allocator)

blt_add_executable(
  NAME backtrace_example
  SOURCES backtrace_example.cpp
  DEPENDS_ON ${example_depends})
list(APPEND umpire_examples backtrace_example)

if (ENABLE_C)
  blt_add_executable(
    NAME allocator_c
    SOURCES allocator.c
    DEPENDS_ON ${example_depends})
  list(APPEND umpire_examples allocator)
endif ()

blt_add_executable(
  NAME vector_allocator
  SOURCES vector_allocator.cpp
  DEPENDS_ON ${example_depends})
list(APPEND umpire_examples vector_allocator)

blt_add_executable(
  NAME size_limiter
  SOURCES size_limiter.cpp
  DEPENDS_ON ${example_depends})
list(APPEND umpire_examples size_limiter)

blt_add_executable(
  NAME use_after_free
  SOURCES use_after_free.cpp
  DEPENDS_ON ${example_depends})
list(APPEND umpire_examples use_after_free)

if (ENABLE_CUDA)
  blt_add_executable(
    NAME device_allocator_example
    SOURCES device-allocator.cpp
    DEPENDS_ON umpire cuda
  )

  set_source_files_properties(
    device-allocator.cpp
    PROPERTIES CUDA_SEPARABLE_COMPILATION On)

  set_target_properties(
    device_allocator_example
    PROPERTIES CUDA_SEPARABLE_COMPILATION On)

  list(APPEND umpire_examples device_allocator_example)

  blt_add_executable(
    NAME multi_device
    SOURCES multi_device.cpp
    DEPENDS_ON umpire cuda)

  set_source_files_properties(
    multi_device.cpp
    PROPERTIES CUDA_SEPARABLE_COMPILATION On)

  set_target_properties(
    multi_device
    PROPERTIES CUDA_SEPARABLE_COMPILATION On)

  list(APPEND umpire_examples multi_device)
endif()

install(TARGETS ${umpire_examples} RUNTIME DESTINATION bin)

add_subdirectory(tutorial)

add_subdirectory(cookbook)

umpire_add_code_checks(PREFIX examples)
