//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_Synchronizer_HPP
#define UMPIRE_Synchronizer_HPP

#include "camp/resource.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/util/MemoryResourceTraits.hpp"

namespace umpire {

class Allocator;

namespace strategy {

class Synchronizer : public AllocationStrategy {
 public:
  Synchronizer(const std::string& name, int id, Allocator allocator, camp::resources::Resource r,
               bool sync_before_alloc = true, bool sync_before_dealloc = true);

  void* allocate(std::size_t bytes) override;
  void deallocate(void* ptr, std::size_t size) override;

  Platform getPlatform() noexcept override;

  MemoryResourceTraits getTraits() const noexcept override;

 private:
  strategy::AllocationStrategy* m_allocator;
  camp::resources::Resource m_resource;
  bool m_sync_before_alloc;
  bool m_sync_before_dealloc;
};

} // end of namespace strategy
} // end of namespace umpire

#endif
