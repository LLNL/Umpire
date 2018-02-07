#ifndef UMPIRE_GenericAllocationStrategyFactory_INL
#define UMPIRE_GenericAllocationStrategyFactory_INL

#include "umpire/strategy/GenericAllocationStrategyFactory.hpp"

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace strategy {

template <typename ALLOC_STRATEGY>
GenericAllocationStrategyFactory<ALLOC_STRATEGY>::GenericAllocationStrategyFactory(const std::string& name):
  m_name(name)
{
}

template <typename ALLOC_STRATEGY>
bool 
GenericAllocationStrategyFactory<ALLOC_STRATEGY>::isValidAllocationStrategyFor(const std::string& name)
{
  return (name.compare(m_name) == 0);
}

template <typename ALLOC_STRATEGY>
std::shared_ptr<AllocationStrategy> 
GenericAllocationStrategyFactory<ALLOC_STRATEGY>::create(
    const std::string& name,
    int id,
    util::AllocatorTraits traits,
    std::vector<std::shared_ptr<AllocationStrategy> > providers)
{
  return std::make_shared<ALLOC_STRATEGY>(name, id, traits, providers);
}

}
}

#endif
