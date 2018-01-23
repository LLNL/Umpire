#include "umpire/op/CudaCopyFromOperation.hpp"

#include <cuda_runtime_api.h>

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace op {

void CudaCopyFromOperation::transform(
    void* src_ptr,
    void* dst_ptr,
    util::AllocationRecord* UMPIRE_UNUSED_ARG(src_allocation),
    util::AllocationRecord* UMPIRE_UNUSED_ARG(dst_allocation),
    size_t length)
{
  cudaError_t error = 
    ::cudaMemcpy(dest_ptr, src_ptr, length, cudaMemcpyDeviceToHost);

  if (error != cudaSuccess) {
    UMPIRE_ERROR("cudaMemcpy( dest_ptr = " << dest_ptr
      << ", src_ptr = " << src_ptr
      << ", length = " << length
      << ", cudaMemcpyDeviceToHost ) failed with error: "
      << cudaGetErrorString(error));
  }
}

} // end of namespace op
} // end of namespace umpire
