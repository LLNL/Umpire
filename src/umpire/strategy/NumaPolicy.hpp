//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
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
          const std::string& name
        , int id
        , Allocator allocator
        , int numa_node);

    void* allocate(std::size_t bytes);
    void deallocate(void* ptr);

    std::size_t getCurrentSize() const noexcept;
    std::size_t getHighWatermark() const noexcept;

    Platform getPlatform() noexcept;

    MemoryResourceTraits getTraits() const noexcept;

    int getNode() const noexcept;

  private:
    strategy::AllocationStrategy* m_allocator;
    Platform m_platform;
    int m_node;
};

} // end of namespace strategy
} // end of namespace umpire

#endif // UMPIRE_NumaPolicy_HPP
