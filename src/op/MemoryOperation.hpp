#ifndef UMPIRE_MemoryOperation_HPP
#define UMPIRE_MemoryOperation_HPP

#include <cstddef>

#include "umpire/util/AllocationRecord.hpp"

namespace umpire {
namespace op {

class MemoryOperation {
  public:
    virtual void transform(
        void* src_ptr,
        void* dst_ptr,
        util::AllocationRecord *src_allocation,
        util::AllocationRecord *dst_allocation,
        size_t length);

    virtual void apply(
        void* src_ptr,
        util::AllocationRecord *src_allocation,
        int val,
        size_t length);
};

} // end of namespace op
} // end of namespace umpire

#endif // UMPIRE_MemoryOperation_HPP
