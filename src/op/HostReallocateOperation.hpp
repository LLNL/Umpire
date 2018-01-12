#ifndef UMPIRE_HostReallocateOperation_HPP
#define UMPIRE_HostReallocateOperation_HPP

#include "MemoryOperation.hpp"

namespace umpire {
namespace op {

class HostReallocateOperation : 
  public MemoryOperation {
 public:
  void transform(
      void** src_ptr,
      void** dst_ptr,
      size_t length);
};

} // end of naemspace op
} // end of namespace umpire

#endif // UMPIRE_HostReallocateOperation_HPP

