//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_MonotonicAllocationStrategy_HPP
#define UMPIRE_MonotonicAllocationStrategy_HPP

#include <memory>
#include <vector>

#include "umpire/Allocator.hpp"
#include "umpire/config.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"

#if defined(UMPIRE_V1_DELEGATE_TO_V2)
#include "umpire/strategy/detail/v1_backed_memory.hpp"
#include "umpire/strategy/monotonic_buffer.hpp"
#endif

namespace umpire {

namespace strategy {

class MonotonicAllocationStrategy : public AllocationStrategy {
 public:
  MonotonicAllocationStrategy(const std::string& name, int id, Allocator allocator, std::size_t capacity);

  ~MonotonicAllocationStrategy();

  void* allocate(std::size_t bytes) override;

  void deallocate(void* ptr, std::size_t size) override;

  std::size_t getCurrentSize() const noexcept override;
  std::size_t getHighWatermark() const noexcept override;

  Platform getPlatform() noexcept override;

  MemoryResourceTraits getTraits() const noexcept override;

 private:
  void* m_block;

  std::size_t m_size;
  std::size_t m_capacity;

  strategy::AllocationStrategy* m_allocator;

#if defined(UMPIRE_V1_DELEGATE_TO_V2)
  // Compile-time-only layout difference (see QuickPool.hpp for the shared
  // rationale). m_block/m_size/m_capacity/m_allocator above remain the
  // source of truth for allocate()/getCurrentSize()/getHighWatermark(): v1's
  // bump-pointer contract has no alignment between successive allocations
  // (unlike v2's monotonic_buffer<Memory>::allocate(), which aligns each
  // offset up to alignof(std::max_align_t) and returns nullptr for
  // zero-byte requests), so those methods are kept fully native rather than
  // delegated, to avoid changing observable pointer-stride and
  // zero-byte-allocation behavior. m_delegate is used only to acquire and
  // release the single backing block from the v1-backed parent (m_block is
  // set to `m_delegate->get_buffer()` after construction), so that the
  // block's lifetime is still managed through the v2 registry/bridge.
  std::unique_ptr<detail::v1_backed_memory> m_v1_backed_parent;
  std::unique_ptr<monotonic_buffer<detail::v1_backed_memory>> m_delegate;
#endif
};

} // end of namespace strategy
} // end of namespace umpire

#endif // UMPIRE_MonotonicAllocationStrategy_HPP
