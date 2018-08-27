//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
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
        Allocator allocator);

    void* allocate(size_t bytes);

    void deallocate(void* ptr);

    long getCurrentSize();
    long getHighWatermark();
    long getActualSize();

    Platform getPlatform();

  private:
    std::shared_ptr<umpire::strategy::AllocationStrategy> m_allocator;

};

} // end of namespace umpire
} // end of namespace strategy

#endif // UMPIRE_AllocationTracker_HPP
