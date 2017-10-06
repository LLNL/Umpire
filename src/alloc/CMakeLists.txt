set(umpire_alloc_headers
  CudaMallocAllocator.hpp
  CudaMallocManagedAllocator.hpp
  MallocAllocator.hpp)

set (umpire_alloc_sources)

set (umpire_alloc_depends)

if (ENABLE_CUDA)
  set (umpire_alloc_headers
    ${umpire_alloc_headers}
    CnmemAllocator.hpp)

  set (umpire_alloc_depends
    ${umpire_alloc_depends}
    umpire_tpl_cnmem)
endif ()

blt_add_library(
  NAME umpire_alloc
  HEADERS ${umpire_alloc_headers}
  SOURCES ${umpire_alloc_sources}
  HEADERS_OUTPUT_SUBDIR umpire/alloc)

target_link_libraries(
  umpire_alloc
  INTERFACE
  ${umpire_alloc_depends})

install(TARGETS
  umpire_alloc
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
