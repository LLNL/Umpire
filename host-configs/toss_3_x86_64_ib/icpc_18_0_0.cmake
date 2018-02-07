set(CMAKE_CXX_COMPILER "/usr/tce/packages/intel/intel-18.0.0/bin/icpc" CACHE PATH "")
set(CMAKE_C_COMPILER "/usr/tce/packages/intel/intel-18.0.0/bin/icc" CACHE PATH "")

set(COMMON_FLAGS "-gxx-name=/usr/tce/packages/gcc/gcc-7.1.0/bin/g++ -std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "${COMMON_FLAGS} -O3 -finline-functions -axCORE-AVX2 -diag-disable cpu-dispatch" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${COMMON_FLAGS} -O3 -g -finline-functions -axCORE-AVX2 -diag-disable cpu-dispatch" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "${COMMON_FLAGS} -O0 -g" CACHE STRING "")
