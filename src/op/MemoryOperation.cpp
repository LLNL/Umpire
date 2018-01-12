#include "umpire/op/MemoryOperation.hpp"

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace op {

void
MemoryOperation::transform(
      void** UMPIRE_UNUSED_ARG(src_ptr),
      void** UMPIRE_UNUSED_ARG(dst_ptr),
      size_t UMPIRE_UNUSED_ARG(length))
{
  UMPIRE_ERROR("MemoryOperation::transform() is not implemented");
}

void
MemoryOperation::apply(
    void** UMPIRE_UNUSED_ARG(src_ptr),
    size_t UMPIRE_UNUSED_ARG(length),
    int UMPIRE_UNUSED_ARG(val))
{
  UMPIRE_ERROR("MemoryOperation::apply() is not implemented");
}



} // end of namespace op
} // end of namespace umpire
