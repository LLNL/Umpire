#ifndef UMPIRE_MemoryOperation_HPP
#define UMPIRE_MemoryOperation_HPP

#include <cstddef>

namespace umpire {
namespace op {

class MemoryOperation {
  public:
  virtual void operator()(
      const void *src_ptr,
      void* dst_ptr,
      size_t length) = 0;
};

} // end of namespace op
} // end of namespace umpire

#endif // UMPIRE_MemoryOperation_HPP
