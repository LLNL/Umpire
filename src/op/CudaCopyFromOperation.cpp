#include "CudaCopyFromOperation.hpp"
#include "../util/AllocationRecord.hpp"

#include <cuda_runtime_api.h>

namespace umpire {
namespace op {

void CudaCopyFromOperation::transform(
    util::AllocationRecord *src_allocation,
    util::AllocationRecord *dest_ptr,
    size_t length)
{
  ::cudaMemcpy(*dest_ptr, *src_allocation, length, cudaMemcpyDeviceToHost);
}

} // end of namespace op
} // end of namespace umpire
