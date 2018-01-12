#ifndef UMPIRE_CudaCopyToOperation_HPP
#define UMPIRE_CudaCopyToOperation_HPP

#include <umpire/util/AllocationRecord.hpp>
#include "MemoryOperation.hpp"
#include "../util/AllocationRecord.hpp"

namespace umpire {
namespace op {

class CudaCopyToOperation : public MemoryOperation {
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

#endif // UMPIRE_CudaCopyToOperation_HPP
