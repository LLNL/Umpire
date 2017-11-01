#ifndef UMPIRE_AllocationStrategyFactory_HPP
#define UMPIRE_AllocationStrategyFactory_HPP

#include "umpire/strategy/AllocationStrategy.hpp"

#include <memory>
#include <string>

namespace umpire {
namespace strategy {

class AllocationStrategyFactory {
  public:
    virtual bool isValidAllocationStrategyFor(const std::string& name) = 0;
    virtual std::shared_ptr<AllocationStrategy> create() = 0;
};

} // end of namespace strategy
} // end of namespace umpire

#endif // UMPIRE_AllocationStrategyFactory_HPP
