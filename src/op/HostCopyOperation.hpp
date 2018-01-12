#ifndef UMPIRE_HostCopyOperation_HPP
#define UMPIRE_HostCopyOperation_HPP

#include "MemoryOperation.hpp"

namespace umpire {
namespace op {

class HostCopyOperation : public MemoryOperation {
 public:
  void transform(
      void** src_ptr,
      void** dst_ptr,
      size_t length);
};

} // end of naemspace op
} //end of namespace umpire

#endif // UMPIRE_HostCopyOperation_HPP
