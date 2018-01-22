#include "umpire/op/CudaCopyToOperation.hpp"

#include <cuda_runtime_api.h>

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace op {

void CudaCopyToOperation::transform(
    void* src_ptr,
    void* dst_ptr,
    umpire::util::AllocationRecord* UMPIRE_UNUSED_ARG(src_allocation),
    umpire::util::AllocationRecord* UMPIRE_UNUSED_ARG(dst_allocation),
    size_t length)
{
  ::cudaMemcpy(
      dst_ptr,
      src_ptr,
      length, 
      cudaMemcpyHostToDevice);
}

} // end of namespace op
} // end of namespace umpire
