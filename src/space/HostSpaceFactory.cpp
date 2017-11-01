#include "umpire/space/HostSpaceFactory.hpp"

#include "umpire/space/MemorySpace.hpp"
#include "umpire/alloc/MallocAllocator.hpp"

namespace umpire {
namespace space {

bool
HostSpaceFactory::isValidAllocationStrategyFor(const std::string& name)
{
  if (name.compare("HOST") == 0) {
    return true;
  } else {
    return false;
  }
}

std::shared_ptr<strategy::AllocationStrategy>
HostSpaceFactory::create()
{
  return std::make_shared<MemorySpace<alloc::MallocAllocator> >();
}

} // end of namespace space
} // end of namespace umpire
