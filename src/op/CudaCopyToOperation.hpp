#ifndef UMPIRE_CudaCopyToOperation_HPP
#define UMPIRE_CudaCopyToOperation_HPP

#include "MemoryOperation.hpp"

namespace umpire {

class CudaCopyToOperation : public MemoryOperation {
 public:
  CudaCopyToOperation();

  void operator()(void *dst_ptr,
      void* src_ptr,
      size_t length);
};

} //end of namespace umpire

#endif // UMPIRE_CudaCopyToOperation_HPP
