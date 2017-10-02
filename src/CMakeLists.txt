set (umpire_headers
  Allocator.hpp
  AllocatorInterface.hpp
  AllocatorFactory.hpp
  AllocatorRegistry.hpp
  ResourceManager.hpp
  Umpire.hpp)

set (umpire_sources
  Allocator.cpp
  AllocatorRegistry.cpp
  ResourceManager.cpp)

if (ENABLE_FORTRAN)
  set (umpire_headers
    ${umpire_headers}
    interface/umpire.h
    shroudrt.hpp
    wrapAllocator.h
    wrapResourceManager.h
    )

  set (umpire_sources
    ${umpire_sources}
    shroudrt.cpp
    wrapAllocator.cpp
    wrapResourceManager.cpp
    wrapfumpire.f
    )
endif ()


blt_add_library(
  NAME umpire
  HEADERS ${umpire_headers}
  SOURCES ${umpire_sources}
  HEADERS_OUTPUT_SUBDIR umpire
  DEPENDS_ON umpire_space umpire_alloc)

configure_file(
  ${PROJECT_SOURCE_DIR}/src/config.hpp.in
  ${PROJECT_BINARY_DIR}/include/umpire/config.hpp)

install(TARGETS
  umpire
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

install(DIRECTORY
  ${CMAKE_BINARY_DIR}/include
  DESTINATION ${CMAKE_INSTALL_PREFIX})

add_subdirectory(space)
add_subdirectory(alloc)
add_subdirectory(op)
add_subdirectory(util)
add_subdirectory(tpl)
if (SHROUD_FOUND)
  add_subdirectory(interface)
endif ()
