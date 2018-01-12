#include "umpire/op/CudaMemsetOperation.hpp"

#include <cuda_runtime_api.h>

namespace umpire {
namespace op {

void
CudaMemsetOperation::apply(
    void** ptr,
    size_t length,
    int value)
{
  ::cudaMemset(*ptr, value, length);
}

} // end of namespace op
} // end of namespace umpire
