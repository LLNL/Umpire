#include "HostCopyOperation.hpp"
#include "../util/AllocationRecord.hpp"

#include <cstring>

#include "umpire/util/AllocationRecord.hpp"

namespace umpire {
namespace op {

void HostCopyOperation::transform(
    void* src_ptr,
    void* dst_ptr,
    util::AllocationRecord *src_allocation,
    util::AllocationRecord *dst_allocation,
    size_t length)
{
  std::memcpy(
      dst_ptr,
      src_ptr,
      length);
}

} // end of namespace op
} // end of namespace umpire
