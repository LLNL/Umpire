#include "CopyOperation.hpp"

#include "umpire/space/HostSpace.hpp"
#include "umpire/space/GpuSpace.hpp"

namespace umpire {

CopyOperation::CopyOperation() {}

void CopyOperation::operator()(void *dest_ptr,
    void* src_ptr,
                             const MemorySpace &source,
                             const MemorySpace &dest)
{

  dest->copyIn(source->copyOut(ptr));

  dest->copy(dest_ptr, src_ptr, /* how to copy */);





}

}
