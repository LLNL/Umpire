#include "HostReallocateOperation.hpp"

#include <cstdlib>

namespace umpire {
namespace op {

void HostReallocateOperation::transform(
    void** src_ptr,
    void** dest_ptr,
    size_t length)
{
  *dest_ptr = ::realloc(*src_ptr, length);
}

} // end of namespace op
} // end of namespace umpire
