#include "umpire/space/UnifiedMemorySpaceFactory.hpp"

#include "umpire/space/MemorySpace.hpp"
#include "umpire/alloc/CudaMallocManagedAllocator.hpp"

namespace umpire {
namespace space {

bool
UnifiedMemorySpaceFactory::isValidAllocationStrategyFor(const std::string& name)
{
  if (name.compare("UM") == 0) {
    return true;
  } else {
    return false;
  }
}

std::shared_ptr<strategy::AllocationStrategy>
UnifiedMemorySpaceFactory::create()
{
  return std::make_shared<space::MemorySpace<alloc::CudaMallocManagedAllocator> >(Platform::cuda);
}

} // end of namespace space
} // end of namespace umpire
