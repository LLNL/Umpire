#ifndef UMPIRE_DefaultAllocationStrategy_HPP
#define UMPIRE_DefaultAllocationStrategy_HPP

#include "umpire/strategy/AllocationStrategy.hpp"

namespace umpire {
namespace strategy {

class DefaultAllocationStrategy :
  public AllocationStrategy
{
  public:
    DefaultAllocationStrategy(std::shared_ptr<AllocationStrategy> allocator);

    void* allocate(size_t bytes);

    /*!
     * \brief Free the memory at ptr.
     *
     * \param ptr Pointer to free.
     */
    void deallocate(void* ptr);

    long getCurrentSize();
    long getHighWatermark();

    Platform getPlatform();

  protected:
    std::shared_ptr<AllocationStrategy> m_allocator;
};

} // end of namespace strategy
} // end of namespace umpire

#endif
