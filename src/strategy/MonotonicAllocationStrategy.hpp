#ifndef UMPIRE_MonotonicAllocationStrategy_HPP
#define UMPIRE_MonotonicAllocationStrategy_HPP

#include <vector>
#include <unordered_map>

#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/util/AllocatorTraits.hpp"

namespace umpire {

namespace strategy {

class MonotonicAllocationStrategy :
  public AllocationStrategy
{
  public:
    MonotonicAllocationStrategy(util::AllocatorTraits traits, std::vector<std::shared_ptr<AllocationStrategy> > providers);

    void* allocate(size_t bytes);
    void deallocate(void* ptr);

    size_t getSize(void* ptr);

    long getCurrentSize();
    long getHighWatermark();

    Platform getPlatform();

  private:
    void* m_block;

    size_t m_size;
    size_t m_capacity;

    std::unordered_map<void*, size_t> m_sizes;

    std::shared_ptr<AllocationStrategy> m_allocator;
};

} // end of namespace strategy
} // end of namespace umpire

#endif // UMPIRE_MonotonicAllocationStrategy_HPP

