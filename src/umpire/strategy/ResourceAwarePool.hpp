//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
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
#include <optional>

#include "camp/camp.hpp"
#include "camp/resource.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/strategy/PoolCoalesceHeuristic.hpp"
#include "umpire/strategy/mixins/AlignedAllocation.hpp"
#include "umpire/util/FixedMallocPool.hpp"
#include "umpire/util/MemoryResourceTraits.hpp"

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

  // Granting the Umpire free function access to private methods below for testing
  friend camp::resources::Resource umpire::get_resource(Allocator a, void* ptr);
  friend std::size_t umpire::get_num_pending(Allocator a);

  /*!
   * \brief If this method is called, it will log a warning message and call allocate with the default Host resource.
   * (Need to call allocate with a Camp resource instead).
   */
  void* allocate(std::size_t bytes) override;

  /*!
   * \brief Allocate memory with the ResourceAwarePool
   *
   * \param bytes The size in bytes for the allocation
   * \param r The Camp resource that will own the memory
   */
  void* allocate_resource(std::size_t bytes, camp::resources::Resource r) override;

  /*!
   * \brief Deallocate memory with the ResourceAwarePool
   *
   * \param ptr A pointer to the memory allocation
   * \param r The Camp resource that owns the memory
   * \param bytes The size in bytes for the allocation
   */
  void deallocate_resource(void* ptr, camp::resources::Resource r, std::size_t size) override;

  /*!
   * \brief Deallocate function will call private getResource function
   * to get the resource associated with the pointer and then call deallocate_resource
   * above.
   */
  void deallocate(void* ptr, std::size_t size) override;

  /*!
   * \brief Release function will first check to see if there are any finished
   * pending chunks and then release both the finished pending chunks and the free chunks.
   * It will only wait for a pending chunk to finish if we are destructing the pool.
   */
  void release() override;

  std::size_t getActualSize() const noexcept override;
  std::size_t getReleasableSize() const noexcept;
  std::size_t getActualHighwaterMark() const noexcept;

  /*!
   * \brief Returns the current size of the pool, rounded up to alignment
   */
  std::size_t getAlignedSize() const noexcept;

  std::size_t getAlignedHighwaterMark() const noexcept;

  Platform getPlatform() noexcept override;
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

  void coalesce() noexcept;
  void do_coalesce(std::size_t suggested_size) noexcept;

  struct Chunk;

 private:
  void do_deallocate(Chunk* chunk, void* ptr) noexcept;

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
  using PendingMap = std::unordered_multimap<std::optional<Resource>, Chunk*>;
  using SizeMap =
      std::multimap<std::size_t, Chunk*, std::less<std::size_t>, pool_allocator<std::pair<const std::size_t, Chunk*>>>;

 protected:
  /*!
   * \brief Get the camp resource associated with a ptr. This function is meant for internal use within the class and
   * for testing.
   *
   * \param ptr The pointer to data allocated with a ResourceAwarePool
   */
  Resource getResource(void* ptr) const;

  /*!
   * \brief Get the number of Pending chunks in the pool. This function is meant for testing.
   */
  std::size_t getNumPending() const noexcept;

 public:
  struct Chunk {
    Chunk(void* ptr, std::size_t s, std::size_t cs, Resource r) : data{ptr}, size{s}, chunk_size{cs}, resource{r}
    {
    }

    void* data{nullptr};
    std::size_t size{0};
    std::size_t chunk_size{0};
    bool free{true};
    Chunk* prev{nullptr};
    Chunk* next{nullptr};
    SizeMap::iterator size_map_it;
    PendingMap::iterator pending_map_it;
    Resource resource;
    Event event;
  };

 private:
  PointerMap m_used_map{};
  SizeMap m_free_map{};
  PendingMap m_pending_map{};

  util::FixedMallocPool m_chunk_pool{sizeof(Chunk)};

  PoolCoalesceHeuristic<ResourceAwarePool> m_should_coalesce;

  const std::size_t m_first_minimum_pool_allocation_size;
  const std::size_t m_next_minimum_pool_allocation_size;

  std::size_t m_total_blocks{0};
  std::size_t m_releasable_blocks{0};
  std::size_t m_actual_bytes{0};
  std::size_t m_aligned_bytes{0};
  std::size_t m_releasable_bytes{0};
  std::size_t m_actual_highwatermark{0};
  std::size_t m_aligned_highwatermark{0};
  bool m_is_destructing{false};
  bool m_is_coalescing{false};
};

std::ostream& operator<<(std::ostream& out, umpire::strategy::PoolCoalesceHeuristic<ResourceAwarePool>&);

inline std::string to_string(PoolCoalesceHeuristic<ResourceAwarePool>&)
{
  return "PoolCoalesceHeuristic<ResourceAwarePool>";
}

} // end of namespace strategy
} // end namespace umpire

#endif // UMPIRE_ResourceAwarePool_HPP
