#ifndef UMPIRE_HostMemsetOperation_HPP
#define UMPIRE_HostMemsetOperation_HPP

#include "umpire/op/MemoryOperation.hpp"

namespace umpire {
namespace op {

class HostMemsetOperation : public MemoryOperation {
 public:
  void apply(
      void* src_ptr,
      util::AllocationRecord* allocation,
      int value,
      size_t length);
};

} // end of naemspace op
} //end of namespace umpire

#endif // UMPIRE_HostMemsetOperation_HPP
