//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_NamingShim_HPP
#define UMPIRE_NamingShim_HPP

#include "umpire/Allocator.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"

namespace umpire {
namespace strategy {

class NamingShim : public AllocationStrategy {
 public:
  NamingShim(const std::string& name, int id, Allocator allocator);

  ~NamingShim();

  void* allocate(std::size_t bytes) override;
  void deallocate(void* ptr, std::size_t size) override;

  Platform getPlatform() noexcept override;

  MemoryResourceTraits getTraits() const noexcept override;

 private:
  std::size_t m_counter;
  strategy::AllocationStrategy* m_allocator;
};

} // end of namespace strategy
} // end namespace umpire

#endif // UMPIRE_NamingShim_HPP
