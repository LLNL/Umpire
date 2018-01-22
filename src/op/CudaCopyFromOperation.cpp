#include "umpire/op/CudaCopyFromOperation.hpp"

#include <cuda_runtime_api.h>

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace op {

void CudaCopyFromOperation::transform(
    void* src_ptr,
    void* dst_ptr,
    util::AllocationRecord* UMPIRE_UNUSED_ARG(src_allocation),
    util::AllocationRecord* UMPIRE_UNUSED_ARG(dst_allocation),
    size_t length)
{
  ::cudaMemcpy(
      dst_ptr,
      src_ptr,
      length, 
      cudaMemcpyDeviceToHost);
}

} // end of namespace op
} // end of namespace umpire
