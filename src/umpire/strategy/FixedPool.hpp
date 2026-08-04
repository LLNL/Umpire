//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_FixedPool_HPP
#define UMPIRE_FixedPool_HPP

#include <cstddef>
#include <memory>
#include <vector>

#include "umpire/Allocator.hpp"
#include "umpire/config.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"

#if defined(UMPIRE_V1_DELEGATE_TO_V2)
#include "umpire/strategy/detail/v1_backed_memory.hpp"
#include "umpire/strategy/fixed_pool.hpp"
#endif

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
  FixedPool(const std::string& name, int id, Allocator allocator, const std::size_t object_bytes,
            const std::size_t objects_per_pool = 64 * sizeof(int) * 8) noexcept;

  ~FixedPool();

  FixedPool(const FixedPool&) = delete;

  void* allocate(std::size_t bytes = 0) override final;
  void deallocate(void* ptr, std::size_t size) override final;

  void release() override final;

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
    Pool(AllocationStrategy* allocation_strategy, const std::size_t object_bytes, const std::size_t objects_per_pool,
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

#if defined(UMPIRE_V1_DELEGATE_TO_V2)
  // Compile-time-only layout difference (see QuickPool.hpp for the shared
  // rationale). The members above remain constructed (unused bookkeeping:
  // m_pool stays empty since newPool()/allocInPool() are never called in
  // this branch) so the class layout difference stays minimal.
  //
  // v2's fixed_pool<Memory> has no equivalent to v1's getHighWatermark()
  // (its public surface only exposes object/pool/free/allocated counts), so
  // the high watermark is tracked natively here, updated after each
  // successful delegated allocate() the same way v1's allocate() updates
  // m_highwatermark (max of running current-bytes).
  //
  // v2's fixed_pool<Memory> also has no equivalent of v1's
  // getActualSize()'s bitmap-overhead accounting (v1's m_actual_bytes sums
  // a `m_avail_bytes` malloc'd availability-bitmap per pool in addition to
  // object storage; v2 has no such bitmap since it uses a std::vector free
  // list instead). getActualSize() is therefore computed natively from
  // m_delegate's pool/object counts using v1's exact formula, rather than
  // delegated directly, so it keeps returning a value reflecting the
  // (documented) v1-specific bitmap overhead.
  std::unique_ptr<detail::v1_backed_memory> m_v1_backed_parent;
  std::unique_ptr<fixed_pool<detail::v1_backed_memory>> m_delegate;
  std::size_t m_native_highwatermark{0};
#endif
};

} // end namespace strategy
} // end namespace umpire

#endif // UMPIRE_FixedPool_HPP
