set (umpire_headers
  Allocator.hpp
  ResourceManager.hpp
  TypedAllocator.hpp
  TypedAllocator.inl
  Umpire.hpp)

set (umpire_sources
  Allocator.cpp
  ResourceManager.cpp)

if (ENABLE_FORTRAN)
  set (umpire_headers
    ${umpire_headers}
    interface/umpire.h
    interface/c_fortran/shroudrt.hpp
    interface/c_fortran/wrapAllocator.h
    interface/c_fortran/wrapResourceManager.h
    )

  set (umpire_sources
    ${umpire_sources}
    interface/c_fortran/shroudrt.cpp
    interface/c_fortran/wrapAllocator.cpp
    interface/c_fortran/wrapResourceManager.cpp
    interface/c_fortran/wrapfumpire.f90
    )

  set_source_files_properties(
    interface/c_fortran/wrapfumpire.f90
    PROPERTIES
    Fortran_FORMAT FREE)
endif ()


blt_add_library(
  NAME umpire
  HEADERS ${umpire_headers}
  SOURCES ${umpire_sources}
  HEADERS_OUTPUT_SUBDIR umpire 
  DEPENDS_ON umpire_resource umpire_strategy umpire_alloc umpire_op)

set(UMPIRE_ENABLE_CUDA ${ENABLE_CUDA})
set(UMPIRE_ENABLE_LOGGING ${ENABLE_LOGGING})
set(UMPIRE_ENABLE_SLIC ${ENABLE_SLIC})
set(UMPIRE_ENABLE_ASSERTS ${ENABLE_ASSERTS})

configure_file(
  ${PROJECT_SOURCE_DIR}/src/config.hpp.in
  ${PROJECT_BINARY_DIR}/include/umpire/config.hpp)

install(TARGETS
  umpire
  EXPORT umpire-targets
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

install(DIRECTORY
  ${CMAKE_BINARY_DIR}/include
  DESTINATION ${CMAKE_INSTALL_PREFIX})

add_subdirectory(resource)
add_subdirectory(alloc)
add_subdirectory(op)
add_subdirectory(util)
add_subdirectory(tpl)
add_subdirectory(strategy)
if (SHROUD_FOUND)
  add_subdirectory(interface)
endif ()
