//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_SizeLimiter_HPP
#define UMPIRE_SizeLimiter_HPP

#include <memory>

#include "umpire/Allocator.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"

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
  SizeLimiter(const std::string& name, int id, Allocator allocator,
              std::size_t size_limit);

  void* allocate(std::size_t bytes) override;
  void deallocate(void* ptr) override;

  Platform getPlatform() noexcept override;

  MemoryResourceTraits getTraits() const noexcept override;

 private:
  strategy::AllocationStrategy* m_allocator;

  std::size_t m_size_limit;
  std::size_t m_total_size;
};

} // end of namespace strategy
} // end namespace umpire

#endif // UMPIRE_SizeLimiter_HPP
