#include "umpire/strategy/AllocationStrategyRegistry.hpp"

#include "umpire/strategy/SimpoolAllocationStrategy.hpp"
#include "umpire/strategy/MonotonicAllocationStrategy.hpp"

#include "umpire/strategy/GenericAllocationStrategyFactory.hpp"

#include "umpire/util/AllocatorTraits.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace strategy {

AllocationStrategyRegistry* AllocationStrategyRegistry::s_allocator_registry_instance = nullptr;

AllocationStrategyRegistry&
AllocationStrategyRegistry::getInstance()
{
  if (!s_allocator_registry_instance) {
    s_allocator_registry_instance = new AllocationStrategyRegistry();
  }

  UMPIRE_LOG(Debug, "() returning " << s_allocator_registry_instance);
  return *s_allocator_registry_instance;
}

AllocationStrategyRegistry::AllocationStrategyRegistry() :
  m_allocator_factories()
{
  UMPIRE_LOG(Debug, "() entered");
  registerAllocationStrategy(
      std::make_shared<GenericAllocationStrategyFactory<SimpoolAllocationStrategy> >("POOL"));

  registerAllocationStrategy(
      std::make_shared<GenericAllocationStrategyFactory<MonotonicAllocationStrategy> >("MONOTONIC"));
  UMPIRE_LOG(Debug, "() leaving");
}

void
AllocationStrategyRegistry::registerAllocationStrategy(std::shared_ptr<AllocationStrategyFactory> factory)
{
  m_allocator_factories.push_back(factory);
}

std::shared_ptr<umpire::strategy::AllocationStrategy>
AllocationStrategyRegistry::makeAllocationStrategy(
    const std::string& name, 
    int id,
    const std::string& strategy,
    util::AllocatorTraits traits, 
    std::vector<std::shared_ptr<AllocationStrategy> > providers)
{
  for (auto allocator_factory : m_allocator_factories) {
    if (allocator_factory->isValidAllocationStrategyFor(strategy)) {
        return allocator_factory->create(name, id, traits, providers);
    }
  }

  UMPIRE_ERROR("Unable to find valid allocation strategy for: " << strategy );
}

} // end of namespace strategy
} // end of namespace umpire
