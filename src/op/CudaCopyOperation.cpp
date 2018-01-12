#include "CudaCopyOperation.hpp"
#include "../util/AllocationRecord.hpp"

#include <cuda_runtime_api.h>
#include <umpire/util/AllocationRecord.hpp>

namespace umpire {
namespace op {

void CudaCopyOperation::transform(
    void* src_ptr,
    void* dst_ptr,
    umpire::util::AllocationRecord *src_allocation,
    umpire::util::AllocationRecord *dst_allocation,
    size_t length)
{
  ::cudaMemcpy(
      dst_ptr,
      src_ptr,
      length, 
      cudaMemcpyDeviceToDevice);
}

} // end of namespace op
} // end of namespace umpire
