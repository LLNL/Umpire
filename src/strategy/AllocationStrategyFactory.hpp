#ifndef UMPIRE_AllocationStrategyFactory_HPP
#define UMPIRE_AllocationStrategyFactory_HPP

#include "umpire/AllocatorFactory.hpp"

namespace umpire {

template <typename Strategy>
class AllocationStrategyFactory :
  public AllocatorFactory {
  public:
    virtual bool isValidAllocatorFor(const std::string& name) = 0;
    virtual std::shared_ptr<AllocatorInterface> create() = 0;
};

} // end of namespace umpire

#endif // UMPIRE_AllocationStrategyFactory_HPP
