//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_SizeLimiter_HPP
#define UMPIRE_SizeLimiter_HPP

#include <memory>

#include "umpire/Allocator.hpp"
#include "umpire/config.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"

#if defined(UMPIRE_V1_DELEGATE_TO_V2)
#include "umpire/strategy/detail/v1_backed_memory.hpp"
#include "umpire/strategy/size_limiter.hpp"
#endif

namespace umpire {
namespace strategy {

/*!
 *
 * \brief An allocator with a limited total size.
 *
 * Using this AllocationStrategy with another can be a good way to limit the
 * total size of allocations made on a particular resource or from a particular
 * context.
 */
class SizeLimiter : public AllocationStrategy {
 public:
  SizeLimiter(const std::string& name, int id, Allocator allocator, std::size_t size_limit);

  void* allocate(std::size_t bytes) override;
  void deallocate(void* ptr, std::size_t size) override;

  Platform getPlatform() noexcept override;

  MemoryResourceTraits getTraits() const noexcept override;

 private:
  strategy::AllocationStrategy* m_allocator;

  std::size_t m_size_limit;
  std::size_t m_total_size;

#if defined(UMPIRE_V1_DELEGATE_TO_V2)
  // Compile-time-only layout difference (matches the UMPIRE_RM_USE_NEW_OPS
  // precedent): when delegation is enabled, allocate()/deallocate() forward
  // to a v2 size_limiter<v1_backed_memory> instead of implementing the limit
  // check natively. m_size_limit/m_total_size above remain unused bookkeeping
  // in this configuration but are kept so the class layout difference stays
  // minimal and getTraits()/getPlatform() logic is untouched.
  std::unique_ptr<detail::v1_backed_memory> m_v1_backed_parent;
  std::unique_ptr<size_limiter<detail::v1_backed_memory>> m_delegate;
#endif
};

} // end of namespace strategy
} // end namespace umpire

#endif // UMPIRE_SizeLimiter_HPP
