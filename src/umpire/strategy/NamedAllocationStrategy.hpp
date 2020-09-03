//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_NamedAllocationStrategy_HPP
#define UMPIRE_NamedAllocationStrategy_HPP

#include "umpire/Allocator.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"

namespace umpire {
namespace strategy {

class NamedAllocationStrategy : public AllocationStrategy {
 public:
  NamedAllocationStrategy(const std::string& name, int id, Allocator allocator);

  void* allocate(std::size_t bytes) override;
  void deallocate(void* ptr) override;

  Platform getPlatform() noexcept override;

  MemoryResourceTraits getTraits() const noexcept override;

 protected:
  strategy::AllocationStrategy* m_allocator;
};

} // end of namespace strategy
} // end of namespace umpire

#endif
