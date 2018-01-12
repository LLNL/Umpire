#ifndef UMPIRE_HostReallocateOperation_HPP
#define UMPIRE_HostReallocateOperation_HPP

#include <umpire/util/AllocationRecord.hpp>
#include "MemoryOperation.hpp"
#include "../util/AllocationRecord.hpp"

namespace umpire {
namespace op {

class HostReallocateOperation : 
  public MemoryOperation {
 public:
  void transform(
      util::AllocationRecord *src_allocation,
      util::AllocationRecord *dst_allocation,
      size_t length);
};

} // end of naemspace op
} // end of namespace umpire

#endif // UMPIRE_HostReallocateOperation_HPP

