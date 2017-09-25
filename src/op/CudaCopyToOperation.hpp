#ifndef UMPIRE_CudaCopyToOperation_HPP
#define UMPIRE_CudaCopyToOperation_HPP

#include "MemoryOperation.hpp"

namespace umpire {
namespace op {

class CudaCopyToOperation : public MemoryOperation {
 public:
  void operator()(
      const void *src_ptr,
      void* dst_ptr,
      size_t length);
};

} // end of namespace op
} // end of namespace umpire

#endif // UMPIRE_CudaCopyToOperation_HPP
