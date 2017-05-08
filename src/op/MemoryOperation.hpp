#ifndef UMPIRE_MemoryOperation_HPP
#define UMPIRE_MemoryOperation_HPP

namespace umpire {

class MemoryOperation {
  virtual void operator()(void *ptr,
                          const MemorySpace &src,
                          const MemorySpace &dest) = 0;
};

} // end of namespace umpire

#endif // UMPIRE_MemoryOperation_HPP
