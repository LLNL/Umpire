#include "umpire/op/HostCopyOperation.hpp"

#include <cstring>

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace op {

void HostCopyOperation::transform(
    void* src_ptr,
    void* dst_ptr,
    util::AllocationRecord* UMPIRE_UNUSED_ARG(src_allocation),
    util::AllocationRecord* UMPIRE_UNUSED_ARG(dst_allocation),
    size_t length)
{
  std::memcpy(
      dst_ptr,
      src_ptr,
      length);
}

} // end of namespace op
} // end of namespace umpire
