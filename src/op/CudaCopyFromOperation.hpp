#ifndef UMPIRE_CudaCopyFromOperation_HPP
#define UMPIRE_CudaCopyFromOperation_HPP

#include "MemoryOperation.hpp"

namespace umpire {

class CudaCopyFromOperation : public MemoryOperation {
 public:
  CudaCopyFromOperation();

  void operator()(void *dst_ptr,
      void* src_ptr,
      size_t length);
};

} //end of namespace umpire

#endif // UMPIRE_CudaCopyFromOperation_HPP
