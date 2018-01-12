#ifndef UMPIRE_CudaMemsetOperation_HPP
#define UMPIRE_CudaMemsetOperation_HPP

#include "MemoryOperation.hpp"

namespace umpire {
namespace op {

class CudaMemsetOperation : public MemoryOperation {
 public:
  void apply(
      void** ptr,
      size_t length,
      int value);
};

} // end of naemspace op
} //end of namespace umpire

#endif // UMPIRE_CudaMemsetOperation_HPP
