#ifndef UMPIRE_CopyOperation_HPP
#define UMPIRE_CopyOperation_HPP

#include "MemoryOperation.hpp"

namespace umpire {

class CopyOperation : public MemoryOperation {
 public:
  CopyOperation();

  void operator()(void *ptr,
                  const MemorySpace &dest);
};

} //end of namespace umpire

#endif // UMPIRE_CopyOperation_HPP
