#include "HostReallocateOperation.hpp"


#include <cstdlib>

#include "umpire/util/AllocationRecord.hpp"
#include "umpire/ResourceManager.hpp"

namespace umpire {
namespace op {

void HostReallocateOperation::transform(
    void* src_ptr,
    void* dst_ptr,
    util::AllocationRecord *src_allocation,
    util::AllocationRecord *dst_allocation,
    size_t length)
{

  ResourceManager::getInstance().deregisterAllocation(src_ptr);

  dst_allocation->m_ptr = ::realloc(src_ptr, length);

  dst_allocation->m_size = length;
  ResourceManager::getInstance().registerAllocation(dst_allocation->m_ptr, dst_allocation);
}

} // end of namespace op
} // end of namespace umpire
