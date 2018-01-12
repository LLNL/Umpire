#include "CudaCopyToOperation.hpp"
#include "../util/AllocationRecord.hpp"

#include <cuda_runtime_api.h>
#include <umpire/util/AllocationRecord.hpp>

namespace umpire {
namespace op {

void CudaCopyToOperation::transform(
    umpire::util::AllocationRecord *src_allocation,
    umpire::util::AllocationRecord *dest_ptr,
    size_t length)
{
  ::cudaMemcpy(*dest_ptr, *src_allocation, length, cudaMemcpyHostToDevice);
}

} // end of namespace op
} // end of namespace umpire
