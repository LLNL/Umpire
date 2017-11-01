#include "umpire/space/DeviceSpaceFactory.hpp"

#include "umpire/space/MemorySpace.hpp"
#include "umpire/alloc/CnmemAllocator.hpp"

namespace umpire {
namespace space {

bool
DeviceSpaceFactory::isValidAllocationStrategyFor(const std::string& name)
{
  if (name.compare("DEVICE") == 0) {
    return true;
  } else {
    return false;
  }
}

std::shared_ptr<strategy::AllocationStrategy>
DeviceSpaceFactory::create()
{
  return std::make_shared<space::MemorySpace<alloc::CnmemAllocator> >(Platform::cuda);
}

} // end of namespace space
} // end of namespace umpire
