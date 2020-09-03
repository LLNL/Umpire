//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_FixedPool_HPP
#define UMPIRE_FixedPool_HPP

#include <cstddef>
#include <vector>

#include "umpire/Allocator.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"

namespace umpire {
namespace strategy {

/*!
 * \brief Pool for fixed size allocations
 *
 * This AllocationStrategy provides an efficient pool for fixed size
 * allocations, and used to quickly allocate and deallocate objects.
 */
class FixedPool : public AllocationStrategy {
 public:
  /*!
   * \brief Constructs a FixedPool.
   *
   * \param name The allocator name for reference later in ResourceManager
   * \param id The allocator id for reference later in ResourceManager
   * \param allocator Used for data allocation. It uses std::malloc
   * for internal tracking of these allocations.
   * \param object_bytes The fixed size (in bytes) for each allocation
   * \param objects_per_pool Number of objects in each sub-pool
   * internally. Performance likely improves if this is large, at
   * the cost of memory usage. This does not have to be a multiple
   * of sizeof(int)*8, but it will also likely improve performance
   * if so.
   */
  FixedPool(const std::string& name, int id, Allocator allocator,
            const std::size_t object_bytes,
            const std::size_t objects_per_pool = 64 * sizeof(int) * 8) noexcept;

  ~FixedPool();

  FixedPool(const FixedPool&) = delete;

  void* allocate(std::size_t bytes = 0) override final;
  void deallocate(void* ptr) override final;

  std::size_t getCurrentSize() const noexcept override final;
  std::size_t getHighWatermark() const noexcept override final;
  std::size_t getActualSize() const noexcept override final;

  Platform getPlatform() noexcept override final;
  MemoryResourceTraits getTraits() const noexcept override final;

  bool pointerIsFromPool(void* ptr) const noexcept;

  std::size_t numPools() const noexcept;

 private:
  struct Pool {
    AllocationStrategy* strategy;
    char* data;
    int* avail;
    std::size_t num_avail;
    Pool(AllocationStrategy* allocation_strategy,
         const std::size_t object_bytes, const std::size_t objects_per_pool,
         const std::size_t avail_bytes);
  };

  void newPool();
  void* allocInPool(Pool& p);

  AllocationStrategy* m_strategy;
  std::size_t m_obj_bytes;
  std::size_t m_obj_per_pool;
  std::size_t m_data_bytes;
  std::size_t m_avail_bytes;
  std::size_t m_current_bytes;
  std::size_t m_actual_bytes;
  std::size_t m_highwatermark;
  std::vector<Pool> m_pool;
  // NOTE: struct Pool lacks a non-trivial destructor. If m_pool is
  // ever reduced in size, then .data and .avail have to be manually
  // deallocated to avoid a memory leak.
};

} // end namespace strategy
} // end namespace umpire

#endif // UMPIRE_FixedPool_HPP
