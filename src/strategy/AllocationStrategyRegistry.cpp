#include "umpire/strategy/AllocationStrategyRegistry.hpp"

#include "umpire/strategy/SimpoolAllocationStrategy.hpp"
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

  return *s_allocator_registry_instance;
}

AllocationStrategyRegistry::AllocationStrategyRegistry() :
  m_allocator_factories()
{
  registerAllocationStrategy(
      std::make_shared<GenericAllocationStrategyFactory<SimpoolAllocationStrategy> >("POOL"));
}

void
AllocationStrategyRegistry::registerAllocationStrategy(std::shared_ptr<AllocationStrategyFactory> factory)
{
  m_allocator_factories.push_back(factory);
}

std::shared_ptr<umpire::strategy::AllocationStrategy>
AllocationStrategyRegistry::makeAllocationStrategy(const std::string& name, util::AllocatorTraits traits, std::vector<std::shared_ptr<AllocationStrategy> > providers)
{
  for (auto allocator_factory : m_allocator_factories) {
    if (allocator_factory->isValidAllocationStrategyFor(name)) {
        return allocator_factory->createWithTraits(traits, providers);
    }
  }

  UMPIRE_ERROR("AllocationStrategy " << name << " not found");
}

} // end of namespace strategy
} // end of namespace umpire
