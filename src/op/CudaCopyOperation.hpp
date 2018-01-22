#ifndef UMPIRE_CudaCopyOperation_HPP
#define UMPIRE_CudaCopyOperation_HPP

#include "umpire/op/MemoryOperation.hpp"

namespace umpire {
namespace op {

class CudaCopyOperation : public MemoryOperation {
 public:
  void transform(
      void* src_ptr,
      void* dst_ptr,
      umpire::util::AllocationRecord *src_allocation,
      umpire::util::AllocationRecord *dst_allocation,
      size_t length);
};

} // end of namespace op
} // end of namespace umpire

#endif // UMPIRE_CudaCopyOperation_HPP
