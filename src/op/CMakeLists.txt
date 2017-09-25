set (umpire_op_headers
  CudaCopyFromOperation.hpp
  CudaCopyToOperation.hpp
  HostCopyOperation.hpp
  MemoryOperation.hpp
  MemoryOperationRegistry.hpp)

set (umpire_op_sources
  CudaCopyFromOperation.cpp
  CudaCopyToOperation.cpp
  HostCopyOperation.cpp
  MemoryOperationRegistry.cpp)

set (umpire_op_depends
  umpire_util)

if (ENABLE_CUDA)
  set (umpire_op_depends
    ${umpire_op_depends}
    cuda_runtime)
endif ()

blt_add_library(
  NAME umpire_op
  HEADERS ${umpire_op_headers}
  SOURCES ${umpire_op_sources}
  HEADERS_OUTPUT_SUBDIR umpire/op
  DEPENDS_ON ${umpire_op_depends}
)

install(TARGETS
  umpire_op
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
