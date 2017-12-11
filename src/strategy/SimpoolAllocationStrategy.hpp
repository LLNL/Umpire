#ifndef UMPIRE_SimPool_HPP
#define UMPIRE_SimPool_HPP

#include <memory>
#include <vector>
#include <unordered_map>

#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/util/AllocatorTraits.hpp"
#include "umpire/util/AllocationRecord.hpp"
#include "umpire/tpl/simpool/DynamicPoolAllocator.hpp"

namespace umpire {
namespace strategy {

class SimPoolAllocationStrategy : public AllocationStrategy
{
  public:
    SimPoolAllocationStrategy(
      util::AllocatorTraits,
      std::vector<std::shared_ptr<AllocationStrategy> > providers);

    void* allocate(size_t bytes);

    void deallocate(void* ptr);

    size_t getSize(void* ptr);

    long getCurrentSize();
    long getHighWatermark();

    Platform getPlatform();

  private:
    DynamicPoolAllocator<>* dpa;

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
