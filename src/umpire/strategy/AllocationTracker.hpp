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
#ifndef UMPIRE_AllocationTracker_HPP
#define UMPIRE_AllocationTracker_HPP

#include <memory>

#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/Allocator.hpp"
#include "umpire/strategy/mixins/Inspector.hpp"

namespace umpire {
namespace strategy {

class AllocationTracker :
  public AllocationStrategy,
  private mixins::Inspector
{
  public:
    AllocationTracker(
        const std::string& name,
        int id,
        Allocator allocator,
        bool own=false) noexcept;

    ~AllocationTracker() override;

    void finalize() override;

    void* allocate(size_t bytes) override;

    void deallocate(void* ptr) override;

    void release() override;

    long getCurrentSize() const noexcept override;
    long getHighWatermark() const noexcept override;
    long getActualSize() const noexcept override;

    Platform getPlatform() noexcept override;

    strategy::AllocationStrategy* getAllocationStrategy();

  private:
    const bool m_owns_allocator;
    strategy::AllocationStrategy* m_allocator;

};

} // end of namespace umpire
} // end of namespace strategy

#endif // UMPIRE_AllocationTracker_HPP
