set (umpire_headers
  Allocator.hpp
  AllocationRecord.hpp
  ResourceManager.hpp
  Umpire.hpp)

set (umpire_sources
  ResourceManager.cpp)

blt_add_library(
  NAME umpire
  HEADERS ${umpire_headers}
  SOURCES ${umpire_sources}
  HEADERS_OUTPUT_SUBDIR umpire 
  DEPENDS_ON umpire_space)

add_subdirectory(space)
add_subdirectory(alloc)
add_subdirectory(op)
add_subdirectory(util)
