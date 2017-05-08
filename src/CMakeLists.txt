set (umpire_headers
  AllocationRecord.hpp
  space/HostSpace.hpp
  alloc/MallocAllocator.hpp
  alloc/MallocAllocator.inl
  alloc/MemoryAllocator.hpp
  space/MemorySpace.hpp
  ResourceManager.hpp
  Umpire.hpp)

set (umpire_sources
  space/HostSpace.cpp
  ResourceManager.cpp)

if (ENABLE_CUDA)
  set(umpire_headers
    ${umpire_headers}
    alloc/CudaMallocAllocator.hpp
    alloc/CudaMallocAllocator.inl
    space/GpuSpace.hpp)
  
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
