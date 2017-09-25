#ifndef UMPIRE_HostCopyOperation_HPP
#define UMPIRE_HostCopyOperation_HPP

#include "MemoryOperation.hpp"

namespace umpire {

class HostCopyOperation : public MemoryOperation {
 public:
  HostCopyOperation();

  void operator()(void *src_ptr,
      void* dst_ptr,
      size_t length);
};

} //end of namespace umpire

#endif // UMPIRE_HostCopyOperation_HPP
