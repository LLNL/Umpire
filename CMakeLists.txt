cmake_minimum_required(VERSION 3.2)

project(Umpire LANGUAGES CXX C Fortran)

set(ENABLE_OPENMP On CACHE Bool "")
set(ENABLE_CUDA On CACHE Bool "")
set(ENABLE_COPY_HEADERS On CACHE Bool "")
set(ENABLE_TESTS On CACHE Bool "")
set(ENABLE_BENCHMARKS On CACHE Bool "")
set(ENABLE_GMOCK On CACHE Bool "")
option(ENABLE_EXAMPLES "Build Umpire examples" On)
option(ENABLE_SLIC "Build Umpire with SLIC logging" Off)

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

if (ENABLE_SLIC)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
  find_package(SLIC REQUIRED)
  blt_register_library( NAME umpire_tpl_slic
                        INCLUDES ${SLIC_INCLUDE_DIR}
                        LIBRARIES ${SLIC_LIBRARY}
                      )

  # set(CMAKE_CXX_FLAGS ${SLIC_DEFINITIONS})
  # set(CMAKE_EXE_LINKER_FLAGS ${SLIC_LIBRARY})
  # include_directories(${SLIC_INCLUDE_DIR})
endif ()

include(cmake/SetupUmpireThirdParty.cmake)

if (ENABLE_CUDA)
  set(CUDA_NVCC_FLAGS -std=c++11)
endif ()

add_subdirectory(src)

if (ENABLE_TESTS)
  set (UMPIRE_TEST_OUTPUT_DIR ${CMAKE_BINARY_DIR}/test)
  add_subdirectory(tests)
endif ()
if (ENABLE_BENCHMARKS)
  set (UMPIRE_BENCHMARK_OUTPUT_DIR ${CMAKE_BINARY_DIR}/benchmark)
  add_subdirectory(benchmarks)
endif ()
if (ENABLE_EXAMPLES)
add_subdirectory(examples)
endif ()

add_subdirectory(docs)
