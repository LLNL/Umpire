#ifndef UMPIRE_UnifiedMemorySpaceFactory_HPP
#define UMPIRE_UnifiedMemorySpaceFactory_HPP

#include "umpire/strategy/AllocationStrategyFactory.hpp"

namespace umpire {
namespace space {


class UnifiedMemorySpaceFactory :
  public strategy::AllocationStrategyFactory
{
  bool isValidAllocationStrategyFor(const std::string& name);
  std::shared_ptr<strategy::AllocationStrategy> create();
};

} // end of namespace space
} // end of namespace umpire

#endif // UMPIRE_UnifiedMemorySpaceFactory_HPP
