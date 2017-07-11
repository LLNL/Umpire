#include "CopyOperation.hpp"

#include "umpire/space/HostSpace.hpp"
#include "umpire/space/GpuSpace.hpp"

namespace umpire {

CopyOperation::CopyOperation() {}

void CopyOperation::operator()(void *dest_ptr,
    void* src_ptr,
    size_t length,
                             const MemorySpace &source,
                             const MemorySpace &dest)
{

  /*
   * Space has:
   * - copy (
   * - copy_in (cudaMemCpyHost2Device)
   * - copy_out (cudaMemCpyDevice2Host)
   */

  if (source == dst)
    source.fast_copy(src_ptr, dest_ptr, len);
  else if (source == DEVCICE)
    source.copy_to //DevcieToHost
  else (/* dst == DEVCIE*/)
    source.copy_from //HostToDevice


  

  dest->copyIn(source->copyOut(ptr));

  dest->copy(dest_ptr, src_ptr, /* how to copy */);





}

}
