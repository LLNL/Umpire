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
#ifndef UMPIRE_NumaPolicy_HPP
#define UMPIRE_NumaPolicy_HPP

#include <vector>

#include "umpire/strategy/AllocationStrategy.hpp"

#include "umpire/Allocator.hpp"

namespace umpire {

namespace strategy {

/*!
 * \brief Use NUMA interface to locate memory to a specific NUMA node.
 *
 * This AllocationStrategy provides a method of ensuring memory sits
 * on a specific NUMA node. This can be used either for optimization,
 * or for moving memory between the host and devices.
 */
class NumaPolicy :
  public AllocationStrategy
{
  public:
    NumaPolicy(
        const std::string& name,
        int id,
        int numa_node,
        Allocator allocator);

    void* allocate(size_t bytes);
    void deallocate(void* ptr);

    long getCurrentSize() const noexcept;
    long getHighWatermark() const noexcept;

    Platform getPlatform() noexcept;

    int getNode() const noexcept;

  private:
    int m_node;
    std::shared_ptr<AllocationStrategy> m_allocator;
};

} // end of namespace strategy
} // end of namespace umpire

#endif // UMPIRE_NumaPolicy_HPP
