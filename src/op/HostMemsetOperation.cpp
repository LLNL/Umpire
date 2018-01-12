#include "umpire/op/HostMemsetOperation.hpp"

#include <cstring>

namespace umpire {
namespace op {

void
HostMemsetOperation::apply(
    void** ptr,
    size_t length,
    int value)
{
  std::memset(*ptr, value, length);
}

} // end of namespace op
} // end of namespace umpire
