#include "HostCopyOperation.hpp"
#include "../util/AllocationRecord.hpp"

#include <cstring>
#include <umpire/util/AllocationRecord.hpp>

namespace umpire {
namespace op {

void HostCopyOperation::transform(
    umpire::util::AllocationRecord *src_allocation,
    umpire::util::AllocationRecord *dest_ptr,
    size_t length)
{
  std::memcpy(*dest_ptr, *src_allocation, length);
}

} // end of namespace op
} // end of namespace umpire
