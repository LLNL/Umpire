//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_NamedAllocationStrategy_HPP
#define UMPIRE_NamedAllocationStrategy_HPP

#include <memory>

#include "umpire/Allocator.hpp"
#include "umpire/config.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"

#if defined(UMPIRE_V1_DELEGATE_TO_V2)
#include "umpire/strategy/detail/v1_backed_memory.hpp"
#include "umpire/strategy/named.hpp"
#endif

namespace umpire {
namespace strategy {

class NamedAllocationStrategy : public AllocationStrategy {
 public:
  NamedAllocationStrategy(const std::string& name, int id, Allocator allocator);

  void* allocate(std::size_t bytes) override;
  void* allocate_named(const std::string& name, std::size_t bytes) override;
  void deallocate(void* ptr, std::size_t size) override;

  Platform getPlatform() noexcept override;

  MemoryResourceTraits getTraits() const noexcept override;

 protected:
  strategy::AllocationStrategy* m_allocator;

#if defined(UMPIRE_V1_DELEGATE_TO_V2)
  // allocate()/deallocate() forward through v2's named<v1_backed_memory>
  // (a pure passthrough decorator, matching this class's own passthrough
  // semantics). allocate_named() has no v2 counterpart -- named<Memory>
  // does not model per-allocation naming -- so it continues to call
  // m_allocator->allocate_named_internal() directly in both configurations.
  std::unique_ptr<detail::v1_backed_memory> m_v1_backed_parent;
  std::unique_ptr<named<detail::v1_backed_memory>> m_delegate;
#endif
};

} // end of namespace strategy
} // end of namespace umpire

#endif
