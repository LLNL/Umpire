#ifndef UMPIRE_SimPool_HPP
#define UMPIRE_SimPool_HPP

#include <memory>
#include <vector>

#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/util/AllocatorTraits.hpp"
#include "umpire/tpl/simpool/DynamicPoolAllocator.hpp"

namespace umpire {

namespace strategy {

struct simPoolAllocator
{
  static inline void* allocate(
      std::shared_ptr<umpire::strategy::AllocationStrategy> ma,
      std::size_t size)
  {
    return ma->allocate(size);
  }
  static inline void deallocate(
      std::shared_ptr<umpire::strategy::AllocationStrategy> ma,
      void *ptr)
  {
    ma->deallocate(ptr);
  }
};

class SimPoolAllocationStrategy : public AllocationStrategy
{
  public:
    SimPoolAllocationStrategy(
      util::AllocatorTraits,
      std::vector<std::shared_ptr<AllocationStrategy> > providers) :
        m_current_size(0),
        m_highwatermark(0)
    {
      m_allocator = providers[0];
      dpa = new DynamicPoolAllocator<simPoolAllocator>((1 << 8), m_allocator);
    }

    void* allocate(size_t bytes) { return dpa->allocate(bytes); }
    void deallocate(void* ptr)   { dpa->deallocate(ptr); }

    long getCurrentSize()        { return m_allocator->getCurrentSize(); }
    long getHighWatermark()      { return m_allocator->getHighWatermark(); }
    size_t getSize(void* ptr)    { return m_allocator->getSize(ptr); }
    Platform getPlatform()       { return m_allocator->getPlatform(); }

  private:
    DynamicPoolAllocator<simPoolAllocator>* dpa;

    void** m_pointers;
    size_t* m_lengths;

    long m_current_size;
    long m_highwatermark;

    size_t m_slots;

    std::shared_ptr<umpire::strategy::AllocationStrategy> m_allocator;
};

} // end of namespace strategy
} // end namespace umpire

#endif // UMPIRE_SimPool_HPP
