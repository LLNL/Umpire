#ifndef UMPIRE_GenericAllocationStrategyFactory_HPP
#define UMPIRE_GenericAllocationStrategyFactory_HPP

#include "umpire/strategy/AllocationStrategyFactory.hpp"

namespace umpire {

namespace strategy {

template <typename ALLOC_STRATEGY>
class GenericAllocationStrategyFactory 
  : public AllocationStrategyFactory {
  public:
    GenericAllocationStrategyFactory(const std::string& name);
    bool isValidAllocationStrategyFor(const std::string& name);
    std::shared_ptr<AllocationStrategy> create();
    std::shared_ptr<AllocationStrategy> createWithTraits(
        const std::string& name,
        util::AllocatorTraits traits,
        std::vector<std::shared_ptr<AllocationStrategy> > providers);
  private:
    std::string m_name;
};

} // end of namespace strategy
} // end of namespace umpire

#include "umpire/strategy/GenericAllocationStrategyFactory.inl"

#endif // UMPIRE_GenericAllocationStrategyFactory_HPP
