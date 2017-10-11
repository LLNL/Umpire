#include "CudaCopyOperation.hpp"

#include <cuda_runtime_api.h>

namespace umpire {
namespace op {

void CudaCopyOperation::operator()(
    const void *src_ptr,
    void* dest_ptr,
    size_t length)
{
  ::cudaMemcpy(dest_ptr, src_ptr, length, cudaMemcpyDeviceToDevice);
}

} // end of namespace op
} // end of namespace umpire
