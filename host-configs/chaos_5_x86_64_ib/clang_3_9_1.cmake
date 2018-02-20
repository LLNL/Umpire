set(CMAKE_CXX_COMPILER "/usr/global/tools/clang/chaos_5_x86_64_ib/clang-3.9.1/bin/clang++" CACHE PATH "")
set(CMAKE_C_COMPILER "/usr/global/tools/clang/chaos_5_x86_64_ib/clang-3.9.1/bin/clang" CACHE PATH "")
set(CMAKE_Fortran_COMPILER "/usr/apps/gnu/4.9.3/bin/gfortran" CACHE PATH "")

include(${CMAKE_CURRENT_LIST_DIR}/cudatoolkit_8_0.cmake)
