set (umpire_space_headers
      CnmemPool.hpp
      CnmemPoolFactory.hpp
      DeviceSpaceFactory.hpp
      HostSpaceFactory.hpp
      MemorySpace.hpp
      MemorySpaceDescriptors.hpp
      MemorySpaceFactory.hpp
      MemorySpaceRegistry.hpp
)

set (umpire_space_sources
      CnmemPool.cpp
      CnmemPoolFactory.cpp
      DeviceSpaceFactory.cpp
      HostSpaceFactory.cpp
      MemorySpace.cpp
      MemorySpaceRegistry.cpp
)

blt_add_library(
  NAME umpire_space
  HEADERS ${umpire_space_headers}
  SOURCES ${umpire_space_sources}
  HEADERS_OUTPUT_SUBDIR umpire/space
  DEPENDS_ON umpire umpire_alloc umpire_util umpire_tpl_cnmem cuda_runtime)

install(TARGETS
  umpire_space
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
