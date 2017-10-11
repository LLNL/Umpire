#include "HostCopyOperation.hpp"

#include <cstring>

namespace umpire {
namespace op {

void HostCopyOperation::operator()(
    const void *src_ptr,
    void* dest_ptr,
    size_t length)
{
  std::memcpy(dest_ptr, src_ptr, length);
}

} // end of namespace op
} // end of namespace umpire
