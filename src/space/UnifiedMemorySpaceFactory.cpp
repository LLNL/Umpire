#include "umpire/space/UnifiedMemorySpaceFactory.hpp"

#include "umpire/alloc/CudaMallocManagedAllocator.hpp"

namespace umpire {
namespace space {

UnifiedMemorySpaceFactory::isValidAllocatorFor(const std::string& name)
{
  if (name.compare("UM") == 0) {
    return true;
  } else {
    return false;
  }
}

std::shared_ptr<AllocatorInterface>
UnifiedMemorySpaceFactory::create()
{
  return std::make_shared<umpire::MemorySpace<alloc::CudaMallocManagedAllocator> >();
}

} // end of namespace space
} // end of namespace umpire
