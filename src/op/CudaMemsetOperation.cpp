#include "umpire/op/CudaMemsetOperation.hpp"

#include <cuda_runtime_api.h>

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace op {

void
CudaMemsetOperation::apply(
    void* src_ptr,
    util::AllocationRecord*  UMPIRE_UNUSED_ARG(allocation),
    int value,
    size_t length)
{
  ::cudaMemset(src_ptr, value, length);
}

} // end of namespace op
} // end of namespace umpire
