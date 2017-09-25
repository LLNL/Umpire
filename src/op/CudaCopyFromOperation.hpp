#ifndef UMPIRE_CudaCopyFromOperation_HPP
#define UMPIRE_CudaCopyFromOperation_HPP

#include "MemoryOperation.hpp"

namespace umpire {
namespace op {

class CudaCopyFromOperation :
  public MemoryOperation {
 public:
  void operator()(
      const void *src_ptr,
      void* dst_ptr,
      size_t length);
};

} // end of namespace op
} //end of namespace umpire

#endif // UMPIRE_CudaCopyFromOperation_HPP
