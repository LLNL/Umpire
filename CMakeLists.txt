cmake_minimum_required(VERSION 3.2)

project(Umpire LANGUAGES CXX)

set(ENABLE_CUDA On)
set(ENABLE_COPY_HEADERS On)

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
add_subdirectory(docs)
