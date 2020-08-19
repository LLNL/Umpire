//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_PoolMap_HPP
#define UMPIRE_PoolMap_HPP

#include <functional>
#include <map>
#include <tuple>
#include <unordered_map>

#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/strategy/mixins/AlignedAllocation.hpp"
#include "umpire/util/MemoryMap.hpp"

namespace umpire {

class Allocator;

namespace util {

class FixedMallocPool;

}

namespace strategy {

class QuickPool : public AllocationStrategy, private mixins::AlignedAllocation {
 public:
  using Pointer = void*;
  using CoalesceHeuristic = std::function<bool(const strategy::QuickPool&)>;

  static CoalesceHeuristic percent_releasable(int percentage);

  /*!
   * \brief Construct a new QuickPool.
   *
   * \param name Name of this instance of the QuickPool
   * \param id Unique identifier for this instance
   * \param allocator Allocation resource that pool uses
   * \param first_minimum_pool_allocation_size Size the pool initially allocates
   * \param next_minimum_pool_allocation_size The minimum size of all future
   * allocations \param alignment Number of bytes with which to align allocation
   * sizes (power-of-2) \param should_coalesce Heuristic for when to perform
   * coalesce operation
   */
  QuickPool(
      const std::string& name, int id, Allocator allocator,
      const std::size_t first_minimum_pool_allocation_size = (512 * 1024 *
                                                              1024),
      const std::size_t next_minimum_pool_allocation_size = (1 * 1024 * 1024),
      const std::size_t alignment = 16,
      CoalesceHeuristic should_coalesce = percent_releasable(100)) noexcept;

  ~QuickPool();

  QuickPool(const QuickPool&) = delete;

  void* allocate(std::size_t bytes) override;
  void deallocate(void* ptr) override;
  void release() override;

  std::size_t getActualSize() const noexcept override;
  std::size_t getReleasableSize() const noexcept;

  Platform getPlatform() noexcept override;

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

  void coalesce() noexcept;

 private:
  struct Chunk;

  template <typename Value>
  class pool_allocator {
   public:
    using value_type = Value;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    pool_allocator() : pool{sizeof(Value)}
    {
    }

    /// BUG: Only required for MSVC
    template <typename U>
    pool_allocator(const pool_allocator<U>& other) : pool{other.pool}
    {
    }

    Value* allocate(std::size_t n)
    {
      return static_cast<Value*>(pool.allocate(n));
    }

    void deallocate(Value* data, std::size_t)
    {
      pool.deallocate(data);
    }

    util::FixedMallocPool pool;
  };

  using PointerMap = std::unordered_map<void*, Chunk*>;
  using SizeMap =
      std::multimap<std::size_t, Chunk*, std::less<std::size_t>,
                    pool_allocator<std::pair<const std::size_t, Chunk*>>>;

  struct Chunk {
    Chunk(void* ptr, std::size_t s, std::size_t cs)
        : data{ptr}, size{s}, chunk_size{cs}
    {
    }

    void* data{nullptr};
    std::size_t size{0};
    std::size_t chunk_size{0};
    bool free{true};
    Chunk* prev{nullptr};
    Chunk* next{nullptr};
    SizeMap::iterator size_map_it;
  };

  PointerMap m_pointer_map{};
  SizeMap m_size_map{};

  util::FixedMallocPool m_chunk_pool{sizeof(Chunk)};

  CoalesceHeuristic m_should_coalesce;

  const std::size_t m_first_minimum_pool_allocation_size;
  const std::size_t m_next_minimum_pool_allocation_size;

  std::size_t m_actual_bytes{0};
  std::size_t m_releasable_bytes{0};
};

} // end of namespace strategy
} // end namespace umpire

#endif // UMPIRE_Pool_HPP
