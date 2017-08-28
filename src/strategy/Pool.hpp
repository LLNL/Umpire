#ifndef UMPIRE_Pool_HPP
#define UMPIRE_Pool_HPP

#include "umpire/strategy/AllocationStrategy.hpp"

#include <memory>

namespace umpire {
namespace strategy {

class Pool :
  public AllocationStrategy
{
  public:
    Pool(std::shared_ptr<umpire::AllocatorInterface>& allocator);

    void* allocate(size_t bytes);
    void deallocate(void* ptr);

  private:
    void init();

    void* m_pointers[32];
    int m_lengths[32];

};

} // end of namespace strategy
} // end namespace umpire

#endif // UMPIRE_Pool_HPP
