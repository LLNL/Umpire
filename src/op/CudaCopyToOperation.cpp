#include "CudaCopyToOperation.hpp"

#include <cuda_runtime_api.h>

namespace umpire {
namespace op {

void CudaCopyToOperation::operator()(
    const void *src_ptr,
    void* dest_ptr,
    size_t length)
{
  ::cudaMemcpy(dest_ptr, src_ptr, length, cudaMemcpyHostToDevice);
}

} // end of namespace op
} // end of namespace umpire
