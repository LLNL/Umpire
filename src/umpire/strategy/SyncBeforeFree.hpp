//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_SyncBeforeFree_HPP
#define UMPIRE_SyncBeforeFree_HPP

#include "camp/resource.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"

namespace umpire {
namespace strategy {

class SyncBeforeFree : public AllocationStrategy {
 public:
  SyncBeforeFree(const std::string& name, int id, Allocator allocator, camp::resources::Resource r);

  void* allocate(std::size_t bytes) override;
  void deallocate(void* ptr, std::size_t size) override;

  Platform getPlatform() noexcept override;

  MemoryResourceTraits getTraits() const noexcept override;

 private:
  strategy::AllocationStrategy* m_allocator;
  camp::resources::Resource m_resource;
};

} // end of namespace strategy
} // end of namespace umpire

#endif