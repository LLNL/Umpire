set (umpire_op_headers
  GenericReallocateOperation.hpp
  HostCopyOperation.hpp
  HostMemsetOperation.hpp
  HostReallocateOperation.hpp
  MemoryOperation.hpp
  MemoryOperationRegistry.hpp)

if (ENABLE_CUDA)
  set (umpire_op_headers
    ${umpire_op_headers}
    CudaCopyOperation.hpp
    CudaCopyFromOperation.hpp
    CudaCopyToOperation.hpp
    CudaMemsetOperation.hpp)
endif ()

set (umpire_op_sources
  GenericReallocateOperation.cpp
  HostCopyOperation.cpp
  HostMemsetOperation.cpp
  HostReallocateOperation.cpp
  MemoryOperation.cpp
  MemoryOperationRegistry.cpp)

if (ENABLE_CUDA)
  set (umpire_op_sources
    ${umpire_op_sources}
    CudaCopyOperation.cpp
    CudaCopyFromOperation.cpp
    CudaCopyToOperation.cpp
    CudaMemsetOperation.cpp)
endif ()

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
