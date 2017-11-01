#ifndef UMPIRE_DeviceSpaceFactory_HPP
#define UMPIRE_DeviceSpaceFactory_HPP

#include "umpire/strategy/AllocationStrategyFactory.hpp"

namespace umpire {
namespace space {


class DeviceSpaceFactory :
  public strategy::AllocationStrategyFactory
{
  bool isValidAllocationStrategyFor(const std::string& name);
  std::shared_ptr<strategy::AllocationStrategy> create();
};

} // end of namespace space
} // end of namespace umpire

#endif // UMPIRE_DeviceSpaceFactory_HPP
