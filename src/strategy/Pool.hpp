#ifndef UMPIRE_Pool_HPP
#define UMPIRE_Pool_HPP

#include "umpire/strategy/AllocationStrategy.hpp"

#include <memory>
#include <vector>

namespace umpire {

  class AllocatorTraits;

namespace strategy {

class Pool :
  public AllocationStrategy
{
  public:
    Pool(AllocatorTraits traits,
      std::vector<std::shared_ptr<AllocationStrategy> > providers);

    void* allocate(size_t bytes);
    void deallocate(void* ptr);

    long getCurrentSize();
    long getHighWatermark();

    size_t getSize(void* ptr);
    Platform getPlatform();

  private:
    void init();

    void** m_pointers;
    int* m_lengths;

    long m_current_size;
    long m_highwatermark;

    size_t m_slots;

    std::shared_ptr<umpire::strategy::AllocationStrategy> m_allocator;
};

} // end of namespace strategy
} // end namespace umpire

#endif // UMPIRE_Pool_HPP
