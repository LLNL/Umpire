#include "umpire/strategy/DefaultAllocationStrategyFactory.hpp"

namespace umpire {
namespace resource {

DefaultAllocationStrategyFactory::DefaultAllocationStrategyFactory()

bool 
isValidAllocationStrategyFor(const std::string& name)
{
}

std::shared_ptr<AllocationStrategy> 
create();

} // end of namespace resource
} // end of namespace umpire
