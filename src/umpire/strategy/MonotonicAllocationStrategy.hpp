//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_MonotonicAllocationStrategy_HPP
#define UMPIRE_MonotonicAllocationStrategy_HPP

#include <vector>

#include "umpire/strategy/AllocationStrategy.hpp"

#include "umpire/Allocator.hpp"

namespace umpire {

namespace strategy {

class MonotonicAllocationStrategy :
  public AllocationStrategy
{
  public:
    MonotonicAllocationStrategy(
        const std::string& name,
        int id,
        std::size_t capacity,
        Allocator allocator);

    ~MonotonicAllocationStrategy();

    void* allocate(std::size_t bytes);

    void deallocate(void* ptr);

    std::size_t getCurrentSize() const noexcept;
    std::size_t getHighWatermark() const noexcept;

    Platform getPlatform() noexcept;

  private:
    void* m_block;

    std::size_t m_size;
    std::size_t m_capacity;

    strategy::AllocationStrategy* m_allocator;
};

} // end of namespace strategy
} // end of namespace umpire

#endif // UMPIRE_MonotonicAllocationStrategy_HPP
