//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_ZeroByteHandler_HPP
#define UMPIRE_ZeroByteHandler_HPP

#include <memory>

#include "umpire/Allocator.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/strategy/FixedPool.hpp"

namespace umpire {
namespace strategy {

class ZeroByteHandler : public AllocationStrategy {
 public:
  ZeroByteHandler(std::unique_ptr<AllocationStrategy>&& allocator) noexcept;

  void* allocate(std::size_t bytes) override;

  void deallocate(void* ptr) override;

  void release() override;

  std::size_t getCurrentSize() const noexcept override;
  std::size_t getHighWatermark() const noexcept override;
  std::size_t getActualSize() const noexcept override;

  Platform getPlatform() noexcept override;

  MemoryResourceTraits getTraits() const noexcept override;

  strategy::AllocationStrategy* getAllocationStrategy();

 private:
  std::unique_ptr<strategy::AllocationStrategy> m_allocator;
  FixedPool* m_zero_byte_pool;
};

} // namespace strategy
} // namespace umpire

#endif // UMPIRE_ZeroByteHandler_HPP
