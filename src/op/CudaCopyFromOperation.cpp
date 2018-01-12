#include "CudaCopyFromOperation.hpp"
#include "../util/AllocationRecord.hpp"

#include <cuda_runtime_api.h>

namespace umpire {
namespace op {

void CudaCopyFromOperation::transform(
    void* src_ptr,
    void* dst_ptr,
    util::AllocationRecord *src_allocation,
    util::AllocationRecord *dst_allocation,
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
