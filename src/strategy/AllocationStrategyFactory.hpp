#ifndef UMPIRE_AllocationStrategyFactory_HPP
#define UMPIRE_AllocationStrategyFactory_HPP

#include <memory>
#include <string>
#include <vector>

#include "umpire/strategy/AllocationStrategy.hpp"

#include "umpire/util/AllocatorTraits.hpp"

namespace umpire {

namespace strategy {

class AllocationStrategyFactory {
  public:
    virtual bool isValidAllocationStrategyFor(const std::string& name) = 0;
    virtual std::shared_ptr<AllocationStrategy> create() = 0;
    virtual std::shared_ptr<AllocationStrategy> createWithTraits(
        util::AllocatorTraits traits,
        std::vector<std::shared_ptr<AllocationStrategy> > providers) = 0;
};

} // end of namespace strategy
} // end of namespace umpire

#endif // UMPIRE_AllocationStrategyFactory_HPP
