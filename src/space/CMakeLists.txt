set (umpire_space_headers
      MemorySpace.hpp
      MemorySpace.inl
      HostSpaceFactory.hpp
)

set (umpire_space_sources
      HostSpaceFactory.cpp
)

set (umpire_space_depends
  umpire_alloc umpire_util)

if (ENABLE_CUDA)
  set (umpire_space_headers
    ${umpire_space_headers}
    DeviceSpaceFactory.hpp
    UnifiedMemorySpaceFactory.hpp)

  set (umpire_space_sources
    ${umpire_space_sources}
    DeviceSpaceFactory.cpp
    UnifiedMemorySpaceFactory.cpp)

  set(umpire_space_depends
    ${umpire_space_depends}
    cuda_runtime)
endif ()

blt_add_library(
  NAME umpire_space
  HEADERS ${umpire_space_headers}
  SOURCES ${umpire_space_sources}
  HEADERS_OUTPUT_SUBDIR umpire/space
  DEPENDS_ON ${umpire_space_depends})

install(TARGETS
  umpire_space
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
