#ifndef UMPIRE_AllocationStrategyRegistry_HPP
#define UMPIRE_AllocationStrategyRegistry_HPP

#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/strategy/AllocationStrategyFactory.hpp"

#include <memory>
#include <list>

namespace umpire {
namespace strategy {

class AllocationStrategyRegistry {
  public:
    static AllocationStrategyRegistry& getInstance();

    std::shared_ptr<umpire::strategy::AllocationStrategy> makeAllocationStrategy(
        const std::string& name);

    void registerAllocationStrategy(std::shared_ptr<AllocationStrategyFactory>&& factory);

  protected:
    AllocationStrategyRegistry();

  private:
    static AllocationStrategyRegistry* s_allocator_registry_instance;

    std::list<std::shared_ptr<AllocationStrategyFactory> > m_allocator_factories;
};

} // end of namespace strategy
} // end of namespace umpire

#endif // UMPIRE_AllocationStrategyRegistry_HPP
