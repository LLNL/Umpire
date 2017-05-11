set(umpire_alloc_headers
    MallocAllocator.hpp
    MallocAllocator.inl
    MemoryAllocator.hpp
    Pool.hpp)

set (umpire_alloc_sources
  Pool.cpp
  )

blt_add_library(
  NAME umpire_alloc
  HEADERS ${umpire_alloc_headers}
  SOURCES ${umpire_alloc_sources}
  HEADERS_OUTPUT_SUBDIR umpire/alloc)
