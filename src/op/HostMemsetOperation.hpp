#ifndef UMPIRE_HostMemsetOperation_HPP
#define UMPIRE_HostMemsetOperation_HPP

#include "MemoryOperation.hpp"

namespace umpire {
namespace op {

class HostMemsetOperation : public MemoryOperation {
 public:
  void apply(
      void** ptr,
      size_t length,
      int value);
};

} // end of naemspace op
} //end of namespace umpire

#endif // UMPIRE_HostMemsetOperation_HPP
