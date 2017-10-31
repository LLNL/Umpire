#include "umpire/space/DeviceSpaceFactory.hpp"

namespace umpire {

bool
template <typename Strategy>
AllocationStrategyFactory<Strategy>::isValidAllocatorFor(const std::string& name)
{
  if (name.compare("POOL") == 0) {
    return true;
  } else {
    return false;
  }
}

template <typename Strategy>
std::shared_ptr<AllocatorInterface>
AllocationStrategyFactory<Strategy>::create()
{
  return std::make_shared<Strategy>();
}

} // end of namespace umpire
