#include "GenericReallocateOperation.hpp"

#include <cstdlib>

#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/ResourceManager.hpp"

#include "umpire/util/AllocationRecord.hpp"

namespace umpire {
namespace op {

void GenericReallocateOperation::transform(
    void* src_ptr,
    void* dst_ptr,
    util::AllocationRecord *src_allocation,
    util::AllocationRecord *dst_allocation,
    size_t length)
{
  // Allocate

  auto allocator = dst_allocation->m_strategy;

  dst_ptr = allocator->allocate(length);

  // Copy
  ResourceManager::getInstance().copy(src_ptr, dst_ptr, length);
  
  // Free
  allocator->deallocate(src_ptr);

  src_allocation->m_ptr = dst_ptr;
}

} // end of namespace op
} // end of namespace umpire
