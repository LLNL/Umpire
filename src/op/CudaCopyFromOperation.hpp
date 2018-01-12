#ifndef UMPIRE_CudaCopyFromOperation_HPP
#define UMPIRE_CudaCopyFromOperation_HPP

#include <umpire/util/AllocationRecord.hpp>
#include "MemoryOperation.hpp"
#include "../util/AllocationRecord.hpp"

namespace umpire {
namespace op {

class CudaCopyFromOperation :
  public MemoryOperation {
 public:
  void transform(
      util::AllocationRecord *src_allocation,
      util::AllocationRecord *dst_allocation,
      size_t length);
};

} // end of namespace op
} //end of namespace umpire

#endif // UMPIRE_CudaCopyFromOperation_HPP
