#ifndef UMPIRE_MemoryOperation_HPP
#define UMPIRE_MemoryOperation_HPP

namespace umpire {

class MemoryOperation {
  virtual void operator()(void *src_ptr,
      void* dst_ptr,
      size_t length) = 0;
};

} // end of namespace umpire

#endif // UMPIRE_MemoryOperation_HPP
