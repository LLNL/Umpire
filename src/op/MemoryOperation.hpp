#ifndef UMPIRE_MemoryOperation_HPP
#define UMPIRE_MemoryOperation_HPP

#include <cstddef>

namespace umpire {
namespace op {

class MemoryOperation {
  public:
    virtual void transform(
        void** src_ptr,
        void** dst_ptr,
        size_t length);

    virtual void apply(
        void** src_ptr,
        size_t length,
        int val);
};

} // end of namespace op
} // end of namespace umpire

#endif // UMPIRE_MemoryOperation_HPP
