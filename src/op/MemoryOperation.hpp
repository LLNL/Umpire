#ifndef UMPIRE_MemoryOperation_HPP
#define UMPIRE_MemoryOperation_HPP

#include <cstddef>

namespace umpire {
namespace op {

class MemoryOperation {
  public:
    virtual void transform(
        util::AllocationRecord *src_allocation,
        util::AllocationRecord *dst_allocation,
        size_t length);

    virtual void apply(util::AllocationRecord *src_allocation, int val, size_t length);
};

} // end of namespace op
} // end of namespace umpire

#endif // UMPIRE_MemoryOperation_HPP
