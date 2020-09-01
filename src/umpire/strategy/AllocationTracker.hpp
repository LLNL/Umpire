//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_AllocationTracker_HPP
#define UMPIRE_AllocationTracker_HPP

#include <memory>

#include "umpire/Allocator.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/strategy/mixins/Inspector.hpp"

namespace umpire {
namespace strategy {

class AllocationTracker : public AllocationStrategy, private mixins::Inspector {
 public:
  AllocationTracker(std::unique_ptr<AllocationStrategy>&& allocator) noexcept;

  void* allocate(std::size_t bytes) override;

  void deallocate(void* ptr) override;

  void release() override;

  std::size_t getCurrentSize() const noexcept override;
  std::size_t getHighWatermark() const noexcept override;
  std::size_t getActualSize() const noexcept override;
  std::size_t getAllocationCount() const noexcept override;

  Platform getPlatform() noexcept override;

  strategy::AllocationStrategy* getAllocationStrategy();

  MemoryResourceTraits getTraits() const noexcept override;

 private:
  std::unique_ptr<strategy::AllocationStrategy> m_allocator;
};

} // namespace strategy
} // namespace umpire

#endif // UMPIRE_AllocationTracker_HPP
