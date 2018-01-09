set (umpire_resource_headers
  MemoryResourceFactory.hpp
  MemoryResourceRegistry.hpp
  DefaultMemoryResource.hpp
  DefaultMemoryResource.inl
  MemoryResource.hpp
  HostResourceFactory.hpp
)

set (umpire_resource_sources
  HostResourceFactory.cpp
  MemoryResource.cpp
  MemoryResourceRegistry.cpp
)

set (umpire_resource_depends
  umpire_alloc umpire_strategy umpire_util)

if (ENABLE_CUDA)
  set (umpire_resource_headers
    ${umpire_resource_headers}
    DeviceResourceFactory.hpp
    UnifiedMemoryResourceFactory.hpp)

  set (umpire_resource_sources
    ${umpire_resource_sources}
    DeviceResourceFactory.cpp
    UnifiedMemoryResourceFactory.cpp)

  set(umpire_resource_depends
    ${umpire_resource_depends}
    cuda_runtime)
endif ()

blt_add_library(
  NAME umpire_resource
  HEADERS ${umpire_resource_headers}
  SOURCES ${umpire_resource_sources}
  HEADERS_OUTPUT_SUBDIR umpire/resource
  DEPENDS_ON ${umpire_resource_depends})

install(TARGETS
  umpire_resource
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
