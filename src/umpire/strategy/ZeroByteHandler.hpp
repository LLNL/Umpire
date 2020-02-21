//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_ZeroByteHandler_HPP
#define UMPIRE_ZeroByteHandler_HPP

#include <memory>

#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/strategy/FixedPool.hpp"

#include "umpire/Allocator.hpp"

namespace umpire {
namespace strategy {

class ZeroByteHandler :
  public AllocationStrategy
{
  public:
    ZeroByteHandler(
        std::unique_ptr<AllocationStrategy>&& allocator) noexcept;

    void* allocate(std::size_t bytes);

    void deallocate(void* ptr);

    void release();

    std::size_t getCurrentSize() const noexcept;
    std::size_t getHighWatermark() const noexcept;
    std::size_t getActualSize() const noexcept;

    Platform getPlatform() noexcept;

    MemoryResourceTraits getTraits() const noexcept;

    strategy::AllocationStrategy* getAllocationStrategy();

  private:
    std::unique_ptr<strategy::AllocationStrategy> m_allocator;
    FixedPool* m_zero_byte_pool;
};

} // end of namespace umpire
} // end of namespace strategy

#endif // UMPIRE_ZeroByteHandler_HPP
