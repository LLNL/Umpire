//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_ResourceAwarePool_HPP
#define UMPIRE_ResourceAwarePool_HPP

#include <functional>
#include <map>
#include <memory>
#include <tuple>
#include <unordered_map>

#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/strategy/PoolCoalesceHeuristic.hpp"
#include "umpire/strategy/mixins/AlignedAllocation.hpp"
#include "umpire/util/FixedMallocPool.hpp"
#include "umpire/util/MemoryResourceTraits.hpp"
#include "camp/camp.hpp"
#include "camp/resource.hpp"

using Resource = camp::resources::Resource;
using Event = camp::resources::Event;

namespace umpire {

class Allocator;

namespace strategy {

class ResourceAwarePool : public AllocationStrategy, private mixins::AlignedAllocation {
 public:
  using Pointer = void*;

  /*!
   * \brief Coalescing Heuristic functions for Percent-Releasable and Blocks-Releasable. Both have
   * the option to reallocate to High Watermark instead of actual size of the pool (actual size is
   * currently the default).
   */
  static PoolCoalesceHeuristic<ResourceAwarePool> percent_releasable(int percentage);
  static PoolCoalesceHeuristic<ResourceAwarePool> percent_releasable_hwm(int percentage);
  static PoolCoalesceHeuristic<ResourceAwarePool> blocks_releasable(std::size_t nblocks);
  static PoolCoalesceHeuristic<ResourceAwarePool> blocks_releasable_hwm(std::size_t nblocks);

  static constexpr std::size_t s_default_first_block_size{512 * 1024 * 1024};
  static constexpr std::size_t s_default_next_block_size{1 * 1024 * 1024};
  static constexpr std::size_t s_default_alignment{16};

  /*!
   * \brief Construct a new ResourceAwarePool.
   *
   * \param name Name of this instance of the ResourceAwarePool
   * \param id Unique identifier for this instance
   * \param allocator Allocation resource that pool uses
   * \param first_minimum_pool_allocation_size Size the pool initially allocates
   * \param next_minimum_pool_allocation_size The minimum size of all future
   * allocations \param alignment Number of bytes with which to align allocation
   * sizes (power-of-2) \param should_coalesce Heuristic for when to perform
   * coalesce operation
   */
  ResourceAwarePool(const std::string& name, int id, Allocator allocator,
            const std::size_t first_minimum_pool_allocation_size = s_default_first_block_size,
            const std::size_t next_minimum_pool_allocation_size = s_default_next_block_size,
            const std::size_t alignment = s_default_alignment,
            PoolCoalesceHeuristic<ResourceAwarePool> should_coalesce = percent_releasable_hwm(100)) noexcept;

  ~ResourceAwarePool();

  ResourceAwarePool(const ResourceAwarePool&) = delete;

  ///////
  void* allocate(std::size_t bytes) override;
  void* allocate_resource(Resource  r, std::size_t bytes) override;
  void deallocate_resource(Resource r, void* ptr, std::size_t size) override;
  void deallocate(void* ptr, std::size_t size) override;
  ///////
  void release() override;

  std::size_t getActualSize() const noexcept override;
  std::size_t getCurrentSize() const noexcept override;
  std::size_t getReleasableSize() const noexcept;
  std::size_t getActualHighwaterMark() const noexcept;

  Platform getPlatform() noexcept override;
  Resource getResource(void* ptr) const noexcept;

  MemoryResourceTraits getTraits() const noexcept override;

  bool tracksMemoryUse() const noexcept override;

  /*!
   * \brief Return the number of memory blocks -- both leased to application
   * and internal free memory -- that the pool holds.
   */
  std::size_t getBlocksInPool() const noexcept;

  /*!
   * \brief Get the largest allocatable number of bytes from pool before
   * the pool will grow.
   *
   * return The largest number of bytes that may be allocated without
   * causing pool growth
   */
  std::size_t getLargestAvailableBlock() noexcept;

  std::size_t getReleasableBlocks() const noexcept;
  std::size_t getTotalBlocks() const noexcept;

  void coalesce(camp::resources::Resource r) noexcept;
  void do_coalesce(camp::resources::Resource r, std::size_t suggested_size) noexcept;

  struct Chunk;

 private:
  void do_deallocate(Chunk* chunk, bool merge_pending_chunk) noexcept;

  template <typename Value>
  class pool_allocator {
   public:
    using value_type = Value;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    pool_allocator() : pool{std::make_shared<util::FixedMallocPool>(sizeof(Value))}
    {
    }

    template <typename U>
    pool_allocator(const pool_allocator<U>& other) : pool{other.pool}
    {
    }

    Value* allocate(std::size_t n)
    {
      return static_cast<Value*>(pool->allocate(n));
    }

    void deallocate(Value* data, std::size_t)
    {
      pool->deallocate(data);
    }

    std::shared_ptr<util::FixedMallocPool> pool;
  };

  using PointerMap = std::unordered_map<void*, Chunk*>;
  ///////
  using PendingMap = std::unordered_map<Chunk*, bool>;
  ///////
  using SizeMap =
      std::multimap<std::size_t, Chunk*, std::less<std::size_t>, pool_allocator<std::pair<const std::size_t, Chunk*>>>;
public:
  struct Chunk {
    Chunk(void* ptr, std::size_t s, std::size_t cs, Resource r) : data{ptr}, size{s}, chunk_size{cs}, m_resource{r}
    {
    }

    void* data{nullptr};
    std::size_t size{0};
    std::size_t chunk_size{0};
    bool free{true};
    Chunk* prev{nullptr};
    Chunk* next{nullptr};
    SizeMap::iterator size_map_it;
    ///////
    Resource m_resource;
    Event m_event;
    bool m_pending{true};
    ///////
  };
private:
  PointerMap m_pointer_map{};
  SizeMap m_size_map{};
  PendingMap m_pending_map{};

  util::FixedMallocPool m_chunk_pool{sizeof(Chunk)};

  PoolCoalesceHeuristic<ResourceAwarePool> m_should_coalesce;

  const std::size_t m_first_minimum_pool_allocation_size;
  const std::size_t m_next_minimum_pool_allocation_size;

  std::size_t m_total_blocks{0};
  std::size_t m_releasable_blocks{0};
  std::size_t m_actual_bytes{0};
  std::size_t m_current_bytes{0};
  std::size_t m_releasable_bytes{0};
  std::size_t m_actual_highwatermark{0};
  bool m_is_destructing{false};
};

std::ostream& operator<<(std::ostream& out, umpire::strategy::PoolCoalesceHeuristic<ResourceAwarePool>&);

inline std::string to_string(PoolCoalesceHeuristic<ResourceAwarePool>&)
{
  return "PoolCoalesceHeuristic<ResourceAwarePool>";
}

} // end of namespace strategy
} // end namespace umpire

#endif // UMPIRE_ResourceAwarePool_HPP
