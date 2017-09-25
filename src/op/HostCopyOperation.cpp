#include "HostCopyOperation.hpp"

namespace umpire {

namespace op {

void HostCopyOperation::operator()(
    void *src_ptr,
    void* dest_ptr,
    size_t length)
{
  std::memcpy(src_ptr, dest_ptr, length);
}

} // end of namespace op
} // end of namespace umpire
