cmake_minimum_required(VERSION 3.2)

project(Umpire LANGUAGES CXX)

set(ENABLE_OPENMP On CACHE Bool "")
set(ENABLE_CUDA Off CACHE Bool "")
set(ENABLE_COPY_HEADERS On CACHE Bool "")
set(ENABLE_TESTS On CACHE Bool "")
set(ENABLE_BENCHMARKS On CACHE Bool "")
option(ENABLE_EXAMPLES "Build Umpire examples" On)

################################
# BLT
################################
if (NOT EXISTS ${PROJECT_SOURCE_DIR}/blt/SetupBLT.cmake)
  message(FATAL_ERROR "\
  The BLT submodule is not present. \
  If in git repository run the following two commands:\n \
  git submodule init\n \
  git submodule update")
endif()

include(blt/SetupBLT.cmake)

add_subdirectory(src)

if (ENABLE_TESTS)
  add_subdirectory(tests)
endif ()
if (ENABLE_BENCHMARKS)
  add_subdirectory(benchmarks)
endif ()
if (ENABLE_EXAMPLES)
add_subdirectory(examples)
endif ()

add_subdirectory(docs)
