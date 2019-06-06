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
#ifndef UMPIRE_ThreadSafeAllocator_HPP
#define UMPIRE_ThreadSafeAllocator_HPP

#include <mutex>

#include "umpire/Allocator.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"

namespace umpire {
namespace strategy {

class ThreadSafeAllocator :
  public AllocationStrategy
{
  public:
    ThreadSafeAllocator(
        const std::string& name,
        int id,
        Allocator allocator);

    void* allocate(size_t bytes);
    void deallocate(void* ptr);

    long getCurrentSize() const noexcept;
    long getHighWatermark() const noexcept;

    Platform getPlatform() noexcept;

  protected:
    strategy::AllocationStrategy* m_allocator;

    std::mutex m_mutex;
};

} // end of namespace strategy
} // end of namespace umpire

#endif // UMPIRE_ThreadSafeAllocator_HPP
