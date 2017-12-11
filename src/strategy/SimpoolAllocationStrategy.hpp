#ifndef UMPIRE_SimpoolAllocationStrategy_HPP
#define UMPIRE_SimpoolAllocationStrategy_HPP

#include <memory>
#include <vector>
#include <unordered_map>

#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/util/AllocatorTraits.hpp"
#include "umpire/util/AllocationRecord.hpp"
#include "umpire/tpl/simpool/DynamicPoolAllocator.hpp"

namespace umpire {
namespace strategy {

class SimpoolAllocationStrategy : public AllocationStrategy
{
  public:
    SimpoolAllocationStrategy(
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

    long m_current_size;
    long m_highwatermark;

    std::shared_ptr<umpire::strategy::AllocationStrategy> m_allocator;
    std::unordered_map<void*, util::AllocationRecord> m_allocations;
};

} // end of namespace strategy
} // end namespace umpire

#endif // UMPIRE_SimpoolAllocationStrategy_HPP
