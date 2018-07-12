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
#ifndef UMPIRE_AllocationAdvisor_HPP
#define UMPIRE_AllocationAdvisor_HPP

#include <memory>

#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/Allocator.hpp"
#include "umpire/op/MemoryOperation.hpp"

namespace umpire {
namespace strategy {

class AllocationAdvisor :
  public AllocationStrategy
{
  public:
      AllocationAdvisor(
        const std::string& name,
        int id,
        Allocator allocator,
        const std::string& advice_operation);

    void* allocate(size_t bytes);
    void deallocate(void* ptr);

    long getCurrentSize();
    long getHighWatermark();

    Platform getPlatform();
  private:
    std::shared_ptr<op::MemoryOperation> m_advice_operation;

    long m_current_size;
    long m_highwatermark;

    std::shared_ptr<umpire::strategy::AllocationStrategy> m_allocator;
};

} // end of namespace strategy
} // end namespace umpire

#endif // UMPIRE_AllocationAdvisor_HPP
