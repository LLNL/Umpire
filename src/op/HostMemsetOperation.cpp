#include "umpire/op/HostMemsetOperation.hpp"

#include <cstring>

namespace umpire {
namespace op {

void
HostMemsetOperation::apply(
    void* src_ptr,
    util::AllocationRecord* allocation,
    int value,
    size_t length)
{
  std::memset(src_ptr, value, length);
}

} // end of namespace op
} // end of namespace umpire
