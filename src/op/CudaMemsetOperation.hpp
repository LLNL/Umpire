#ifndef UMPIRE_CudaMemsetOperation_HPP
#define UMPIRE_CudaMemsetOperation_HPP

#include "umpire/op/MemoryOperation.hpp"

namespace umpire {
namespace op {

class CudaMemsetOperation : public MemoryOperation {
 public:
  void apply(
      void* src_ptr,
      util::AllocationRecord* ptr,
      int value,
      size_t length);
};

} // end of naemspace op
} //end of namespace umpire

#endif // UMPIRE_CudaMemsetOperation_HPP
