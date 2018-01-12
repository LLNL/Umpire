#include "umpire/op/CudaMemsetOperation.hpp"

#include <cuda_runtime_api.h>

namespace umpire {
namespace op {

void
CudaMemsetOperation::apply(
    void* src_ptr,
    util::AllocationRecord* allocation,
    int value,
    size_t length)
{
  ::cudaMemset(src_ptr, value, length);
}

} // end of namespace op
} // end of namespace umpire
