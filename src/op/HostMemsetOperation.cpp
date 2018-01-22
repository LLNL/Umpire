#include "umpire/op/HostMemsetOperation.hpp"

#include <cstring>

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace op {

void
HostMemsetOperation::apply(
    void* src_ptr,
    util::AllocationRecord* UMPIRE_UNUSED_ARG(allocation),
    int value,
    size_t length)
{
  std::memset(src_ptr, value, length);
}

} // end of namespace op
} // end of namespace umpire
