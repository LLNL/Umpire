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
#ifndef UMPIRE_DynamicPool_HPP
#define UMPIRE_DynamicPool_HPP

#include <memory>
#include <vector>

#include "umpire/strategy/AllocationStrategy.hpp"

#include "umpire/Allocator.hpp"

#include "umpire/tpl/simpool/DynamicSizePool.hpp"

namespace umpire {
namespace strategy {

class DynamicPool : public AllocationStrategy
{
  public:
    DynamicPool(
        const std::string& name,
        int id,
        Allocator allocator,
        size_t min_alloc_size = 1 << 8);

    void* allocate(size_t bytes);

    void deallocate(void* ptr);

    long getCurrentSize();
    long getHighWatermark();

    Platform getPlatform();

  private:
    DynamicSizePool<>* dpa;

    long m_current_size;
    long m_highwatermark;

    std::shared_ptr<umpire::strategy::AllocationStrategy> m_allocator;
};

} // end of namespace strategy
} // end namespace umpire

#endif // UMPIRE_DynamicPool_HPP
