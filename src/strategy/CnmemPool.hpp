#ifndef UMPIRE_CnmemPool_HPP_
#define UMPIRE_CnmemPool_HPP_

#include "umpire/strategy/AllocationStrategy.hpp"

namespace umpire {
namespace strategy {

class CnmemPool 
  : public AllocationStrategy
{
  public: 
    CnmemPool(std::shared_ptr<umpire::Allocator>& alloc);

    virtual void* allocate(size_t bytes);
    virtual void deallocate(void* ptr);
};

} // end of namespace space
} // end of namespace umpire

#endif // UMPIRE_CnmemPool_HPP_
