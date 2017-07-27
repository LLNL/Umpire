set(umpire_alloc_headers
  CudaMallocAllocator.hpp
  CudaMallocAllocator.inl
  MallocAllocator.hpp
  MallocAllocator.inl
  MemoryAllocator.hpp
  Pool.hpp
  Pool.inl
  PosixMemalignAllocator.hpp
  PosixMemalignAllocator.inl)

set (umpire_alloc_sources)

blt_add_library(
  NAME umpire_alloc
  HEADERS ${umpire_alloc_headers}
  SOURCES ${umpire_alloc_sources}
  HEADERS_OUTPUT_SUBDIR umpire/alloc)

install(TARGETS
  umpire_alloc
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
