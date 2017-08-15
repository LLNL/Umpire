#include "umpire/strategy/AllocationStrategy.hpp"

namespace umpire {
namespace strategy {
  
AllocationStrategy::AllocationStrategy(std::shared_ptr<umpire::Allocator>& alloc)
  m_allocator(alloc)
{
}

} // end of namespace strategy
} // end of namespace umpire
