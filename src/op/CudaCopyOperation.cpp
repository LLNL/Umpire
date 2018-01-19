#include "CudaCopyOperation.hpp"
#include "umpire/util/Macros.hpp"

#include <cuda_runtime_api.h>

namespace umpire {
namespace op {

void CudaCopyOperation::operator()(
    const void *src_ptr,
    void* dest_ptr,
    size_t length)
{
  cudaError_t error = ::cudaMemcpy(dest_ptr, src_ptr, length, cudaMemcpyDeviceToDevice);

  if (error != cudaSuccess) {
    UMPIRE_ERROR("cudaMemcpy( dest_ptr = " << dest_ptr
      << ", src_ptr = " << src_ptr
      << ", length = " << length
      << ", cudaMemcpyDeviceToDevice ) failed with error: " << cudaGetErrorString(error));
  }
}

} // end of namespace op
} // end of namespace umpire
