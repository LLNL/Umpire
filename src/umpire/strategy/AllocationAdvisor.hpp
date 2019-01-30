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

/*!
 *
 * \brief Applies the given MemoryOperation to every allocation.
 *
 * This AllocationStrategy is designed to be used with the following
 * operations:
 *
 * - op::CudaAdviseAccessedByOperation
 * - op::CudaAdvisePreferredLocationOperation
 * - op::CudaAdviseReadMostlyOperation
 *
 * Using this AllocationStrategy when combined with a pool like DynamicPool is
 * a good way to mitigate the overhead of applying the memory advice.
 */
class AllocationAdvisor :
  public AllocationStrategy
{
  public:
      AllocationAdvisor(
        const std::string& name,
        int id,
        Allocator allocator,
        const std::string& advice_operation);

      AllocationAdvisor(
        const std::string& name,
        int id,
        Allocator allocator,
        const std::string& advice_operation,
        Allocator accessing_allocator);

    void* allocate(size_t bytes);
    void deallocate(void* ptr);

    long getCurrentSize() const noexcept;
    long getHighWatermark() const noexcept;

    Platform getPlatform() noexcept;
  private:
    std::shared_ptr<op::MemoryOperation> m_advice_operation;

    std::shared_ptr<umpire::strategy::AllocationStrategy> m_allocator;

    int m_device;
};

} // end of namespace strategy
} // end namespace umpire

#endif // UMPIRE_AllocationAdvisor_HPP
