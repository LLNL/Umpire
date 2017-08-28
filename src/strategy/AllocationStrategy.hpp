#ifndef UMPIRE_AllocationStrategy_HPP
#define UMPIRE_AllocationStrategy_HPP

#include "umpire/AllocatorInterface.hpp"

namespace umpire {
namespace strategy {

class AllocationStrategy :
  public umpire::AllocatorInterface
{
  public:
    AllocationStrategy(std::shared_ptr<umpire::AllocatorInterface>& alloc);

    virtual void* allocate(size_t bytes) = 0;;
    virtual void deallocate(void* ptr) = 0;;

  protected:
    std::shared_ptr<umpire::Allocatior> m_allocator;
}

} // end of namespace strategy
} // end of namespace umpire

#endif // UMPIRE_AllocationStrategy_HPP
