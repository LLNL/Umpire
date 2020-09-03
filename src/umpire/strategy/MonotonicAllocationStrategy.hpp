//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_MonotonicAllocationStrategy_HPP
#define UMPIRE_MonotonicAllocationStrategy_HPP

#include <vector>

#include "umpire/Allocator.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"

namespace umpire {

namespace strategy {

class MonotonicAllocationStrategy : public AllocationStrategy {
 public:
  MonotonicAllocationStrategy(const std::string& name, int id,
                              Allocator allocator, std::size_t capacity);

  ~MonotonicAllocationStrategy();

  void* allocate(std::size_t bytes) override;

  void deallocate(void* ptr) override;

  std::size_t getCurrentSize() const noexcept override;
  std::size_t getHighWatermark() const noexcept override;

  Platform getPlatform() noexcept override;

  MemoryResourceTraits getTraits() const noexcept override;

 private:
  void* m_block;

  std::size_t m_size;
  std::size_t m_capacity;

  strategy::AllocationStrategy* m_allocator;
};

} // end of namespace strategy
} // end of namespace umpire

#endif // UMPIRE_MonotonicAllocationStrategy_HPP
