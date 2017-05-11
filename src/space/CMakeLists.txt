set (umpire_space_headers
      HostSpaceFactory.hpp
      MemorySpace.hpp
      MemorySpaceDescriptors.hpp
      MemorySpaceFactory.hpp
      MemorySpaceRegistry.hpp
)

set (umpire_space_sources
      HostSpaceFactory.cpp
      MemorySpace.cpp
      MemorySpaceRegistry.cpp
)

blt_add_library(
  NAME umpire_space
  HEADERS ${umpire_space_headers}
  SOURCES ${umpire_space_sources}
  HEADERS_OUTPUT_SUBDIR umpire/space
  DEPENDS_ON umpire_alloc umpire_util)

