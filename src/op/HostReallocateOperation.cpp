#include "HostReallocateOperation.hpp"
#include "../util/AllocationRecord.hpp"

#include <cstdlib>
#include <umpire/util/AllocationRecord.hpp>

namespace umpire {
namespace op {

void HostReallocateOperation::transform(
    util::AllocationRecord *src_allocation,
    util::AllocationRecord *dest_ptr,
    size_t length)
{
  *dest_ptr = ::realloc(*src_allocation, length);
}

} // end of namespace op
} // end of namespace umpire
