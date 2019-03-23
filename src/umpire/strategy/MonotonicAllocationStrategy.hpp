//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// Created by David Beckingsale, david@llnl.gov
// LLNL-CODE-747640
//
// All rights reserved.
//
// This file is part of Umpire.
//
// For details, see https://github.com/LLNL/Umpire
// Please also see the LICENSE file for MIT license.
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
        size_t capacity,
        Allocator allocator);

    ~MonotonicAllocationStrategy() override;

    void finalize() override;

    void* allocate(size_t bytes) override;
    void deallocate(void* ptr) override;

    long getCurrentSize() const noexcept override;
    long getHighWatermark() const noexcept override;

    Platform getPlatform() noexcept override;

  private:
    void free();

    void* m_block;

    size_t m_size;
    size_t m_capacity;

    strategy::AllocationStrategy* m_allocator;
};

} // end of namespace strategy
} // end of namespace umpire

#endif // UMPIRE_MonotonicAllocationStrategy_HPP
