#ifndef UMPIRE_DefaultAllocationStrategyFactory_HPP
#define UMPIRE_DefaultAllocationStrategyFactory_HPP

#include "umpire/strategy/AllocationStrategyFactory.hpp"

#include <memory>
#include <string>

namespace umpire {
namespace strategy {

class DefaultAllocationStrategyFactory :
  public AllocationStrategyFactory {
  public:
    bool isValidAllocationStrategyFor(const std::string& name);
    std::shared_ptr<AllocationStrategy> create();
};

} // end of namespace strategy
} // end of namespace umpire

#endif // UMPIRE_DefaultAllocationStrategyFactory_HPP
