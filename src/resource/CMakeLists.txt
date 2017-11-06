set (umpire_space_headers
  MemoryResourceFactory.hpp
  MemoryResourceRegistry.hpp
  DefaultMemoryResource.hpp
  DefaultMemoryResource.inl
  MemoryResource.hpp
  HostResourceFactory.hpp
)

set (umpire_space_sources
  HostResourceFactory.cpp
  MemoryResourceRegistry.cpp
)

set (umpire_space_depends
  umpire_alloc umpire_strategy umpire_util)

if (ENABLE_CUDA)
  set (umpire_space_headers
    ${umpire_space_headers}
    DeviceResourceFactory.hpp
    UnifiedMemoryResourceFactory.hpp)

  set (umpire_space_sources
    ${umpire_space_sources}
    DeviceResourceFactory.cpp
    UnifiedMemoryResourceFactory.cpp)

  set(umpire_space_depends
    ${umpire_space_depends}
    cuda_runtime)
endif ()

blt_add_library(
  NAME umpire_space
  HEADERS ${umpire_space_headers}
  SOURCES ${umpire_space_sources}
  HEADERS_OUTPUT_SUBDIR umpire/resource
  DEPENDS_ON ${umpire_space_depends})

install(TARGETS
  umpire_space
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
