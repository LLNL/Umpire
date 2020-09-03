//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_NumaPolicy_HPP
#define UMPIRE_NumaPolicy_HPP

#include <vector>

#include "umpire/Allocator.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"

namespace umpire {

namespace strategy {

/*!
 * \brief Use NUMA interface to locate memory to a specific NUMA node.
 *
 * This AllocationStrategy provides a method of ensuring memory sits
 * on a specific NUMA node. This can be used either for optimization,
 * or for moving memory between the host and devices.
 */
class NumaPolicy : public AllocationStrategy {
 public:
  NumaPolicy(const std::string& name, int id, Allocator allocator,
             int numa_node);

  void* allocate(std::size_t bytes) override;
  void deallocate(void* ptr) override;

  Platform getPlatform() noexcept override;

  MemoryResourceTraits getTraits() const noexcept override;

  int getNode() const noexcept;

 private:
  strategy::AllocationStrategy* m_allocator;
  Platform m_platform;
  int m_node;
};

} // end of namespace strategy
} // end of namespace umpire

#endif // UMPIRE_NumaPolicy_HPP
