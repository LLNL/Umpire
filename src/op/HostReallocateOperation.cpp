#include "HostReallocateOperation.hpp"
#include "../util/AllocationRecord.hpp"

#include <cstdlib>
#include <umpire/util/AllocationRecord.hpp>

namespace umpire {
namespace op {

void HostReallocateOperation::transform(
    void* src_ptr,
    void* dst_ptr,
    util::AllocationRecord *src_allocation,
    util::AllocationRecord *dst_allocation,
    size_t length)
{
  dst_allocation->m_ptr = ::realloc(src_ptr, length);
}

} // end of namespace op
} // end of namespace umpire
