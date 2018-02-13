set(CMAKE_CXX_COMPILER "/usr/local/bin/icpc-17.0.174" CACHE PATH "")
set(CMAKE_C_COMPILER "/usr/local/bin/icc-17.0.174" CACHE PATH "")
set(CMAKE_Fortran_COMPILER "/usr/local/bin/ifort-17.0.174" CACHE PATH "")

set(COMMON_FLAGS "-gnu-prefix=/usr/apps/gnu/4.9.3/bin/ -Wl,-rpath,/usr/apps/gnu/4.9.3/lib64 -std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "${COMMON_FLAGS} -O3 -march=native -ansi-alias" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${COMMON_FLAGS} -O3 -g -march=native -ansi-alias" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "${COMMON_FLAGS} -O0 -g" CACHE STRING "")

include(${CMAKE_CURRENT_LIST_DIR}/cudatoolkit_8_0.cmake)
