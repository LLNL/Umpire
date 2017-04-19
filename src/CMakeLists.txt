set (umpire_headers
  AllocationRecord.hpp
  HostSpace.hpp
  MallocAllocator.hpp
  MallocAllocator.inl
  MemoryAllocator.hpp
  MemorySpace.hpp
  ResourceManager.hpp
  Umpire.hpp)

set (umpire_sources
  HostSpace.cpp
  ResourceManager.cpp)

if (ENABLE_CUDA)
  set(umpire_headers
    ${umpire_headers}
    CudaMallocAllocator.hpp
    CudaMallocAllocator.inl
    GpuSpace.hpp)
  
  set(umpire_sources
    ${umpire_sources}
    GpuSpace.cpp)

  set(umpire_depends
    ${umpire_depends}
    cuda_runtime)
endif ()

blt_add_library(
  NAME umpire
  HEADERS ${umpire_headers}
  SOURCES ${umpire_sources}
  HEADERS_OUTPUT_SUBDIR umpire
  DEPENDS_ON ${umpire_depends})

add_subdirectory(tests)
add_subdirectory(benchmarks)
